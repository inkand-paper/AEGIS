import asyncio
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from fakes import ScriptedModel, call_tools, plan_json, say
from pydantic import BaseModel, ConfigDict

from aegis.controller import (
    TERMINAL,
    Controller,
    InvalidTransition,
    TaskNotFound,
    apply_transition,
)
from aegis.memory import EventLog
from aegis.model_client import ModelTransportError
from aegis.models import Budget, BudgetUsage, EventType, Task, TaskState
from aegis.tools import default_registry
from aegis.tools.base import Tool, ToolContext, ToolError, ToolRegistry

# ---------------------------------------------------------------- helpers


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Done(BaseModel):
    note: str


class FlakyTool(Tool[NoArgs, Done]):
    """Fails the first time it runs, then works: a transient failure."""

    name = "flaky"
    description = "A tool that fails once."
    arg_model = NoArgs
    result_model = Done
    side_effects = "none"
    requires_network = False
    idempotent = True

    def __init__(self) -> None:
        self.runs = 0

    async def execute(self, args: NoArgs, ctx: ToolContext) -> Done:
        self.runs += 1
        if self.runs == 1:
            raise ToolError("temporary glitch, please try again")
        return Done(note="worked")


class CancellingTool(Tool[NoArgs, Done]):
    """Asks the controller to cancel the task while it is running."""

    name = "canceller"
    description = "Requests cancellation."
    arg_model = NoArgs
    result_model = Done
    side_effects = "none"
    requires_network = False
    idempotent = True

    def __init__(self, holder: dict[str, Any]) -> None:
        self._holder = holder

    async def execute(self, args: NoArgs, ctx: ToolContext) -> Done:
        self._holder["controller"].cancel(ctx.task_id)
        return Done(note="cancel requested")


@pytest.fixture
def events(tmp_path: Path) -> EventLog:
    return EventLog(tmp_path / "aegis.db")


def make(
    tmp_path: Path,
    events: EventLog,
    model: ScriptedModel,
    registry: ToolRegistry | None = None,
    **kwargs: Any,
) -> Controller:
    return Controller(
        model,
        registry or default_registry(),
        events,
        workspaces_dir=tmp_path / "workspaces",
        **kwargs,
    )


def run(controller: Controller, task_id: str) -> Task:
    return asyncio.run(controller.run(task_id))


def task_transitions(events: EventLog, task_id: str) -> list[str]:
    changes = [
        e.payload
        for e in events.events(task_id, EventType.STATE_CHANGED)
        if e.payload["scope"] == "task"
    ]
    return [f"{c['from']}->{c['to']}" for c in changes]


def assert_audit_trail_is_consistent(events: EventLog, task_id: str) -> None:
    """P5: every TOOL_CALL has exactly one TOOL_RESULT."""
    calls = events.events(task_id, EventType.TOOL_CALL)
    results = events.events(task_id, EventType.TOOL_RESULT)
    assert sorted(c.correlation_id or "" for c in calls) == sorted(
        r.correlation_id or "" for r in results
    )
    assert events.unmatched_tool_calls(task_id) == []


def assert_ended_with(events: EventLog, task_id: str, state: str) -> None:
    last = events.events(task_id)[-1]
    assert last.type is EventType.STATE_CHANGED
    assert last.payload["scope"] == "task"
    assert last.payload["to"] == state


def usage_matches_events(events: EventLog, task: Task) -> None:
    rebuilt = events.recompute_usage(task.id)
    assert rebuilt.model_calls == task.usage.model_calls
    assert rebuilt.tool_calls == task.usage.tool_calls
    assert rebuilt.tokens_total == task.usage.tokens_total
    assert rebuilt.dollars == task.usage.dollars


# ---------------------------------------------------------------- exit criterion 1: 3-step task


def test_three_step_task_completes_with_a_verification_result(
    tmp_path: Path, events: EventLog
) -> None:
    notes = tmp_path / "hello.txt"
    notes.write_text("hello world", encoding="utf-8")
    model = ScriptedModel(
        say(plan_json("Multiply 17 by 23", "Read hello.txt", "Combine the results")),
        call_tools(("calculator", {"expression": "17 * 23"})),
        say("The product is 391."),
        call_tools(("fs_read", {"path": "hello.txt"})),
        say("The file says: hello world"),
        call_tools(("calculator", {"expression": "391 + 9"})),
        say("Final answer: 400, and the file says hello world."),
    )
    controller = make(tmp_path, events, model)
    task = controller.create_task(
        "Multiply 17 by 23, read hello.txt and combine them",
        acceptance=["result_contains:391", "file_read:hello.txt", "answer_contains:400"],
        attach=[notes],
    )
    assert (task.workspace_path / "hello.txt").read_text(encoding="utf-8") == "hello world"

    finished = run(controller, task.id)

    assert finished.state is TaskState.COMPLETED
    assert finished.last_error is None
    assert finished.verification_result_id is not None
    verification = events.events(task.id, EventType.VERIFICATION)[0]
    assert verification.payload["id"] == finished.verification_result_id
    assert verification.payload["passed"] is True
    assert all(check["passed"] for check in verification.payload["checks"])
    assert task_transitions(events, task.id) == [
        "created->running",
        "running->verifying",
        "verifying->completed",
    ]
    snapshot = controller.snapshot(task.id)
    assert [s["status"] for s in snapshot["plan"]["steps"]] == ["done", "done", "done"]
    assert all(s["evidence_ids"] for s in snapshot["plan"]["steps"])
    assert "400" in snapshot["final_answer"]
    assert (finished.usage.model_calls, finished.usage.tool_calls) == (7, 3)
    assert finished.usage.tokens_total == 700
    assert finished.workspace_path.parent == (tmp_path / "workspaces").resolve()
    assert_audit_trail_is_consistent(events, task.id)
    usage_matches_events(events, finished)
    assert_ended_with(events, task.id, "completed")


def test_tool_results_are_sent_back_to_the_model_as_tool_messages(
    tmp_path: Path, events: EventLog
) -> None:
    model = ScriptedModel(
        say(plan_json("Multiply")),
        call_tools(("calculator", {"expression": "17 * 23"})),
        say("391"),
    )
    controller = make(tmp_path, events, model)
    task = controller.create_task("Multiply 17 by 23")
    run(controller, task.id)

    first_turn = model.calls[1]
    assert first_turn[0].role == "system"
    assert "Multiply 17 by 23" in (first_turn[1].content or "")
    second_turn = model.calls[2]
    assert second_turn[-2].role == "assistant"
    assert second_turn[-1].role == "tool"
    assert second_turn[-1].tool_call_id == second_turn[-2].tool_calls[0].id
    assert '"value": 391' in (second_turn[-1].content or "")


# ---------------------------------------------------------------- exit criterion 2: recovery


def test_a_transient_tool_failure_is_recovered_from(tmp_path: Path, events: EventLog) -> None:
    registry = default_registry()
    registry.register(FlakyTool())
    model = ScriptedModel(
        say(plan_json("Use the flaky tool")),
        call_tools(("flaky", {})),
        call_tools(("flaky", {})),
        say("It worked on the second try."),
    )
    controller = make(tmp_path, events, model, registry)
    task = controller.create_task("Use the flaky tool", acceptance=["tool_used:flaky"])
    finished = run(controller, task.id)

    assert finished.state is TaskState.COMPLETED
    outcomes = [e.payload["outcome"] for e in events.events(task.id, EventType.TOOL_RESULT)]
    assert outcomes == ["tool_error", "ok"]
    assert "temporary glitch" in (model.calls[2][-1].content or "")  # the model saw the error
    assert_audit_trail_is_consistent(events, task.id)


def test_a_policy_violation_is_reported_but_never_counts_as_evidence(
    tmp_path: Path, events: EventLog
) -> None:
    notes = tmp_path / "hello.txt"
    notes.write_text("hello world", encoding="utf-8")
    model = ScriptedModel(
        say(plan_json("Read the file")),
        call_tools(("fs_read", {"path": "../../etc/passwd"})),
        call_tools(("fs_read", {"path": "hello.txt"})),
        say("It says hello world."),
    )
    controller = make(tmp_path, events, model)
    task = controller.create_task(
        "Read hello.txt", acceptance=["file_read:hello.txt"], attach=[notes]
    )
    finished = run(controller, task.id)

    assert finished.state is TaskState.COMPLETED
    outcomes = [e.payload["outcome"] for e in events.events(task.id, EventType.TOOL_RESULT)]
    assert outcomes == ["policy_violation", "ok"]
    assert len(events.events(task.id, EventType.EVIDENCE_ADDED)) == 1  # only the good read
    assert "policy_violation" in (model.calls[2][-1].content or "")


# ---------------------------------------------------------------- exit criterion 3: budgets


def looping_model() -> ScriptedModel:
    """A model that never finishes: every reply is another tool call. 100 tokens, $0.10 each."""
    return ScriptedModel(
        say(plan_json("Loop forever"), tokens=100, dollars="0.10"),
        *[
            call_tools(("calculator", {"expression": "1 + 1"}), tokens=100, dollars="0.10")
            for _ in range(10)
        ],
    )


@pytest.mark.parametrize(
    ("limit", "value", "reason", "tool_calls_executed"),
    [
        ("max_tool_calls", 2, "tool_calls 2/2", 2),
        ("max_model_calls", 3, "model_calls 3/3", 1),
        ("max_tokens_total", 250, "tokens_total 300/250", 1),
        ("max_dollars", Decimal("0.25"), "dollars 0.30/0.25", 1),
    ],
)
def test_exceeding_a_budget_fails_the_task(
    tmp_path: Path,
    events: EventLog,
    limit: str,
    value: Any,
    reason: str,
    tool_calls_executed: int,
) -> None:
    controller = make(tmp_path, events, looping_model())
    task = controller.create_task("Loop", budget=Budget(**{limit: value}))
    finished = run(controller, task.id)

    assert finished.state is TaskState.FAILED
    assert finished.last_error == f"BudgetExceeded: {reason}"
    exceeded = events.events(task.id, EventType.BUDGET_EXCEEDED)
    assert [e.payload["reason"] for e in exceeded] == [reason]
    assert len(events.events(task.id, EventType.TOOL_CALL)) == tool_calls_executed
    assert_audit_trail_is_consistent(events, task.id)
    assert_ended_with(events, task.id, "failed")
    usage_matches_events(events, finished)


def test_the_wall_clock_budget_is_enforced(tmp_path: Path, events: EventLog) -> None:
    def slow_plan(messages: Any) -> Any:
        time.sleep(1.1)
        return say(plan_json("Anything"))

    controller = make(tmp_path, events, ScriptedModel(slow_plan))
    task = controller.create_task("Be slow", budget=Budget(max_wall_clock_s=1))
    finished = run(controller, task.id)
    assert finished.state is TaskState.FAILED
    assert (finished.last_error or "").startswith("BudgetExceeded: wall_clock_s")


# ---------------------------------------------------------------- exit criterion 4: cancellation


def test_cancelling_during_a_model_call_stops_the_task(tmp_path: Path, events: EventLog) -> None:
    holder: dict[str, Any] = {}

    def cancel_then_propose_a_tool(messages: Any) -> Any:
        holder["controller"].cancel(holder["task_id"])
        return call_tools(("calculator", {"expression": "1 + 1"}))

    model = ScriptedModel(say(plan_json("One step")), cancel_then_propose_a_tool)
    controller = make(tmp_path, events, model)
    holder["controller"] = controller
    task = controller.create_task("Cancel me")
    holder["task_id"] = task.id

    finished = run(controller, task.id)

    assert finished.state is TaskState.CANCELLED
    assert events.events(task.id, EventType.TOOL_CALL) == []  # the proposed call never ran
    assert_ended_with(events, task.id, "cancelled")
    count = len(events.events(task.id))
    assert run(controller, task.id).state is TaskState.CANCELLED  # nothing more happens
    controller.cancel(task.id)
    assert len(events.events(task.id)) == count


def test_cancelling_during_a_tool_call_finishes_that_call_then_stops(
    tmp_path: Path, events: EventLog
) -> None:
    holder: dict[str, Any] = {}
    registry = default_registry()
    registry.register(CancellingTool(holder))
    model = ScriptedModel(say(plan_json("One step")), call_tools(("canceller", {})))
    controller = make(tmp_path, events, model, registry)
    holder["controller"] = controller
    task = controller.create_task("Cancel me")

    finished = run(controller, task.id)

    assert finished.state is TaskState.CANCELLED
    assert len(events.events(task.id, EventType.TOOL_CALL)) == 1
    assert_audit_trail_is_consistent(events, task.id)  # the running call was not left dangling
    assert_ended_with(events, task.id, "cancelled")
    assert len(model.calls) == 2  # no further model call after the cancel


def test_cancelling_before_the_task_starts(tmp_path: Path, events: EventLog) -> None:
    model = ScriptedModel()  # any model call would raise
    controller = make(tmp_path, events, model)
    task = controller.create_task("Never mind")
    controller.cancel(task.id)
    assert controller.get_task(task.id).state is TaskState.CANCELLED
    count = len(events.events(task.id))
    assert run(controller, task.id).state is TaskState.CANCELLED
    assert len(events.events(task.id)) == count
    assert model.calls == []


# ---------------------------------------------------------------- failure paths


def test_a_step_that_never_finishes_fails_after_its_attempts(
    tmp_path: Path, events: EventLog
) -> None:
    model = ScriptedModel(
        say(plan_json("Never finish")),
        *[call_tools(("calculator", {"expression": "1 + 1"})) for _ in range(4)],
    )
    controller = make(tmp_path, events, model, max_turns_per_step=2)
    task = controller.create_task("Loop")
    finished = run(controller, task.id)

    assert finished.state is TaskState.FAILED
    assert (finished.last_error or "").startswith("StepFailed")
    step = controller.snapshot(task.id)["plan"]["steps"][0]
    assert (step["status"], step["attempts"]) == ("failed", 2)
    assert "previous attempt at this step failed" in (model.calls[3][-1].content or "")


def test_a_task_that_never_used_a_tool_fails_verification(
    tmp_path: Path, events: EventLog
) -> None:
    model = ScriptedModel(say(plan_json("Just answer")), say("Done, no tools needed."))
    controller = make(tmp_path, events, model)
    task = controller.create_task("Say something")
    finished = run(controller, task.id)
    assert finished.state is TaskState.FAILED
    assert "VerificationFailed" in (finished.last_error or "")
    assert "tool_evidence" in (finished.last_error or "")
    assert finished.verification_result_id is None
    assert_ended_with(events, task.id, "failed")


def test_an_unmet_acceptance_criterion_fails_verification(
    tmp_path: Path, events: EventLog
) -> None:
    model = ScriptedModel(
        say(plan_json("Calculate")),
        call_tools(("calculator", {"expression": "17 * 23"})),
        say("The answer is 391"),
    )
    controller = make(tmp_path, events, model)
    task = controller.create_task("Calculate", acceptance=["result_contains:999"])
    finished = run(controller, task.id)
    assert finished.state is TaskState.FAILED
    assert "criterion:result_contains:999" in (finished.last_error or "")


def test_an_invalid_plan_is_retried_once_with_feedback(tmp_path: Path, events: EventLog) -> None:
    model = ScriptedModel(
        say("I think we should just start."),
        say(plan_json("Calculate")),
        call_tools(("calculator", {"expression": "2 + 2"})),
        say("4"),
    )
    controller = make(tmp_path, events, model)
    task = controller.create_task("Add 2 and 2")
    assert run(controller, task.id).state is TaskState.COMPLETED
    assert "rejected" in (model.calls[1][-1].content or "")


def test_two_invalid_plans_fail_the_task(tmp_path: Path, events: EventLog) -> None:
    model = ScriptedModel(say("no plan"), say("still no plan"))
    controller = make(tmp_path, events, model)
    task = controller.create_task("Anything")
    finished = run(controller, task.id)
    assert finished.state is TaskState.FAILED
    assert (finished.last_error or "").startswith("PlanInvalid")


def test_a_model_outage_fails_the_task_and_counts_the_retries(
    tmp_path: Path, events: EventLog
) -> None:
    model = ScriptedModel(
        say(plan_json("Calculate")),
        ModelTransportError("HTTP 503: unavailable", status_code=503, attempts=3),
    )
    controller = make(tmp_path, events, model)
    task = controller.create_task("Calculate")
    finished = run(controller, task.id)
    assert finished.state is TaskState.FAILED
    assert (finished.last_error or "").startswith("ModelError")
    assert finished.usage.model_calls == 1 + 3  # each failed HTTP attempt counts
    usage_matches_events(events, finished)


# ---------------------------------------------------------------- state machine and invariants


def make_task(state: TaskState) -> Task:
    now = datetime.now(UTC)
    return Task(
        id="t",
        objective="o",
        state=state,
        budget=Budget(),
        usage=BudgetUsage(),
        workspace_path=Path("w"),
        created_at=now,
        updated_at=now,
    )


@pytest.mark.parametrize("terminal", sorted(TERMINAL))
@pytest.mark.parametrize("target", [TaskState.RUNNING, TaskState.VERIFYING, TaskState.CREATED])
def test_terminal_states_are_final(terminal: TaskState, target: TaskState) -> None:
    task = make_task(terminal)
    with pytest.raises(InvalidTransition):
        apply_transition(task, target, error="x")
    assert task.state is terminal


@pytest.mark.parametrize(
    ("start", "target"),
    [
        (TaskState.CREATED, TaskState.VERIFYING),
        (TaskState.CREATED, TaskState.COMPLETED),
        (TaskState.RUNNING, TaskState.COMPLETED),
    ],
)
def test_skipping_states_is_rejected(start: TaskState, target: TaskState) -> None:
    with pytest.raises(InvalidTransition):
        apply_transition(make_task(start), target, error="x")


def test_completed_requires_a_verification_result() -> None:
    task = make_task(TaskState.VERIFYING)
    with pytest.raises(InvalidTransition, match="verification"):
        apply_transition(task, TaskState.COMPLETED)
    task.verification_result_id = "v1"
    apply_transition(task, TaskState.COMPLETED)
    assert task.state is TaskState.COMPLETED


def test_failed_requires_an_error_message() -> None:
    task = make_task(TaskState.RUNNING)
    with pytest.raises(InvalidTransition, match="error"):
        apply_transition(task, TaskState.FAILED)
    apply_transition(task, TaskState.FAILED, error="BudgetExceeded: x")
    assert task.last_error == "BudgetExceeded: x"


# ---------------------------------------------------------------- tasks, checkpoints, workspaces


def test_a_new_controller_can_read_a_finished_task_from_the_checkpoint(
    tmp_path: Path, events: EventLog
) -> None:
    model = ScriptedModel(
        say(plan_json("Calculate")),
        call_tools(("calculator", {"expression": "2 * 21"})),
        say("42"),
    )
    first = make(tmp_path, events, model)
    task = first.create_task("Calculate", acceptance=["result_contains:42"])
    run(first, task.id)

    second = make(tmp_path, events, ScriptedModel())
    reloaded = second.get_task(task.id)
    assert reloaded.state is TaskState.COMPLETED
    assert reloaded.verification_result_id is not None
    assert reloaded.acceptance == ["result_contains:42"]
    count = len(events.events(task.id))
    assert run(second, task.id).state is TaskState.COMPLETED  # terminal: nothing re-runs
    assert len(events.events(task.id)) == count


def test_unknown_task_ids_are_reported(tmp_path: Path, events: EventLog) -> None:
    controller = make(tmp_path, events, ScriptedModel())
    with pytest.raises(TaskNotFound):
        controller.get_task("nope")


def test_create_task_validates_its_input(tmp_path: Path, events: EventLog) -> None:
    controller = make(tmp_path, events, ScriptedModel())
    one = tmp_path / "a" / "same.txt"
    two = tmp_path / "b" / "same.txt"
    for path in (one, two):
        path.parent.mkdir()
        path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        controller.create_task("   ")
    with pytest.raises(ValueError, match="criterion"):
        controller.create_task("x", acceptance=["bogus:1"])
    with pytest.raises(ValueError, match="not a file"):
        controller.create_task("x", attach=[tmp_path / "missing.txt"])
    with pytest.raises(ValueError, match="same file name"):
        controller.create_task("x", attach=[one, two])


def test_every_task_gets_its_own_workspace_inside_the_workspaces_folder(
    tmp_path: Path, events: EventLog
) -> None:
    controller = make(tmp_path, events, ScriptedModel())
    first = controller.create_task("one")
    second = controller.create_task("two")
    root = (tmp_path / "workspaces").resolve()
    assert first.workspace_path != second.workspace_path
    assert first.workspace_path.is_relative_to(root)
    assert second.workspace_path.is_relative_to(root)
    assert first.workspace_path.is_dir()
