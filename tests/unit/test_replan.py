"""Replanning (design doc section 10.2): a failed plan is revised, not retried blindly."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fakes import ScriptedModel, call_tools, plan_json, say
from pydantic import BaseModel, ConfigDict

from aegis.controller import Controller
from aegis.memory import EventLog
from aegis.models import Budget, EventType, Plan, StepStatus, Task, TaskState
from aegis.planner import MAX_STEPS, build_plan, merge_revision
from aegis.tools import default_registry
from aegis.tools.base import Tool, ToolContext, ToolError, ToolRegistry

CALC = ("calculator", {"expression": "1 + 1"})


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str = "x"


class Out(BaseModel):
    ok: bool


class NeverWorks(Tool[Args, Out]):
    """A tool whose failure repeats: the same call can never succeed."""

    name = "never_works"
    description = "Always fails."
    arg_model = Args
    result_model = Out
    side_effects = "none"
    requires_network = False
    idempotent = True
    repeatable_after_failure = False

    def __init__(self) -> None:
        self.runs = 0

    async def execute(self, args: Args, ctx: ToolContext) -> Out:
        self.runs += 1
        raise ToolError(f"nothing matches {args.target!r}")


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


def stuck_step_script() -> list[Any]:
    """Step 1 works, step 2 never gives a result (two turns per attempt, two attempts)."""
    return [
        say(plan_json("Add numbers", "Stuck step")),
        call_tools(CALC),
        say("2"),
        *[call_tools(CALC) for _ in range(4)],
    ]


def test_a_failed_step_is_replanned_and_the_task_completes(
    tmp_path: Path, events: EventLog
) -> None:
    model = ScriptedModel(
        *stuck_step_script(), say(plan_json("Report the sum")), say("The sum is 2")
    )
    controller = make(tmp_path, events, model, max_turns_per_step=2)
    task = controller.create_task("Add")
    finished = run(controller, task.id)

    assert finished.state is TaskState.COMPLETED
    revised = events.events(task.id, EventType.PLAN_REVISED)
    assert len(revised) == 1
    assert revised[0].payload["trigger"] == "step_failed"
    assert revised[0].payload["replan"] == 1

    plan = Plan.model_validate(revised[0].payload["plan"])
    first = Plan.model_validate(events.events(task.id, EventType.PLAN_CREATED)[0].payload["plan"])
    assert plan.id != first.id
    assert plan.revision == 2
    assert [s.status for s in plan.steps] == [StepStatus.DONE, StepStatus.PENDING]
    assert plan.steps[0].id == first.steps[0].id  # the finished step is kept, not redone
    assert plan.steps[1].depends_on == [plan.steps[0].id]

    # the revising model was told what was done and what went wrong
    prompt = next(c for c in model.calls if "Steps already done" in (c[-1].content or ""))
    text = prompt[-1].content or ""
    assert "Add numbers -> 2" in text
    assert "step_failed" in text
    assert "Stuck step" in text


def test_a_failed_verification_is_replanned(tmp_path: Path, events: EventLog) -> None:
    model = ScriptedModel(
        say(plan_json("Multiply")),
        call_tools(("calculator", {"expression": "17 * 23"})),
        say("The answer is 391"),
        say(plan_json("Multiply the right numbers")),
        call_tools(("calculator", {"expression": "27 * 37"})),
        say("The answer is 999"),
    )
    controller = make(tmp_path, events, model)
    task = controller.create_task("Multiply", acceptance=["result_contains:999"])
    finished = run(controller, task.id)

    assert finished.state is TaskState.COMPLETED
    revised = events.events(task.id, EventType.PLAN_REVISED)
    assert [e.payload["trigger"] for e in revised] == ["verification_failed"]
    states = [
        f"{e.payload['from']}->{e.payload['to']}"
        for e in events.events(task.id, EventType.STATE_CHANGED)
        if e.payload["scope"] == "task"
    ]
    assert states == [
        "created->running",
        "running->verifying",
        "verifying->running",
        "running->verifying",
        "verifying->completed",
    ]


def test_the_replan_budget_stops_a_task_that_keeps_failing(
    tmp_path: Path, events: EventLog
) -> None:
    attempt = [call_tools(("calculator", {"expression": "17 * 23"})), say("391")]
    model = ScriptedModel(
        say(plan_json("Multiply")),
        *attempt,
        say(plan_json("Multiply again")),
        *attempt,
    )
    controller = make(tmp_path, events, model)
    task = controller.create_task(
        "Multiply", acceptance=["result_contains:999"], budget=Budget(max_replans=1)
    )
    finished = run(controller, task.id)

    assert finished.state is TaskState.FAILED
    assert (finished.last_error or "").startswith("ReplanBudgetExceeded")
    assert "result_contains:999" in (finished.last_error or "")
    assert len(events.events(task.id, EventType.PLAN_REVISED)) == 1


def test_replanning_can_be_switched_off(tmp_path: Path, events: EventLog) -> None:
    model = ScriptedModel(*stuck_step_script())
    controller = make(tmp_path, events, model, max_turns_per_step=2)
    task = controller.create_task("Add", budget=Budget(max_replans=0))
    finished = run(controller, task.id)

    assert finished.state is TaskState.FAILED
    assert (finished.last_error or "").startswith("StepFailed")
    assert events.events(task.id, EventType.PLAN_REVISED) == []


def test_a_bad_revised_plan_fails_the_task_clearly(tmp_path: Path, events: EventLog) -> None:
    model = ScriptedModel(*stuck_step_script(), say("not json"), say("still not json"))
    controller = make(tmp_path, events, model, max_turns_per_step=2)
    task = controller.create_task("Add")
    finished = run(controller, task.id)

    assert finished.state is TaskState.FAILED
    assert (finished.last_error or "").startswith("PlanInvalid")


def test_a_resume_uses_the_latest_plan_even_if_the_checkpoint_is_behind(
    tmp_path: Path, events: EventLog
) -> None:
    model = ScriptedModel(*stuck_step_script(), say(plan_json("Report the sum")), say("2"))
    controller = make(tmp_path, events, model, max_turns_per_step=2)
    task = controller.create_task("Add")
    run(controller, task.id)
    revised = Plan.model_validate(events.events(task.id, EventType.PLAN_REVISED)[0].payload["plan"])

    snapshot = events.load_checkpoint(task.id)
    assert snapshot is not None
    snapshot["plan"] = events.events(task.id, EventType.PLAN_CREATED)[0].payload["plan"]
    events.save_checkpoint(task.id, snapshot)  # as if the crash came between event and checkpoint

    fresh = make(tmp_path, events, ScriptedModel())
    loaded = fresh._load_run(task.id)
    assert loaded.replans == 1  # counted from the log, not from the checkpoint
    assert loaded.plan is not None and loaded.plan.id != revised.id
    fresh._recover(loaded)
    assert loaded.plan is not None and loaded.plan.id == revised.id


def test_an_identical_failed_call_is_not_run_again(tmp_path: Path, events: EventLog) -> None:
    registry = default_registry()
    tool = NeverWorks()
    registry.register(tool)
    same = ("never_works", {"target": "Submit"})
    model = ScriptedModel(
        say(plan_json("Click it")),
        call_tools(same),
        call_tools(same),  # refused: it failed a moment ago
        call_tools(("never_works", {"target": "Send"})),  # a different call is allowed
        say("Could not do it."),
    )
    controller = make(tmp_path, events, model, registry)
    task = controller.create_task("Click", budget=Budget(max_replans=0))
    run(controller, task.id)

    outcomes = [e.payload["outcome"] for e in events.events(task.id, EventType.TOOL_RESULT)]
    assert outcomes == ["tool_error", "repeat_blocked", "tool_error"]
    assert tool.runs == 2
    assert events.unmatched_tool_calls(task.id) == []  # the refused call is still on the record
    blocked = next(
        c for c in model.calls[3] if c.role == "tool" and "already failed" in (c.content or "")
    )
    assert "different arguments" in (blocked.content or "")


def test_a_repeatable_tool_may_be_called_again_after_a_failure(
    tmp_path: Path, events: EventLog
) -> None:
    bad = ("calculator", {"expression": "1 / 0"})
    model = ScriptedModel(
        say(plan_json("Divide")), call_tools(bad), call_tools(bad), say("undefined")
    )
    controller = make(tmp_path, events, model)
    task = controller.create_task("Divide", budget=Budget(max_replans=0))
    run(controller, task.id)
    outcomes = [e.payload["outcome"] for e in events.events(task.id, EventType.TOOL_RESULT)]
    assert "repeat_blocked" not in outcomes


def test_merge_keeps_done_steps_and_chains_the_new_ones() -> None:
    old = build_plan(
        "t", {"steps": [{"description": "a"}, {"description": "b"}, {"description": "c"}]}
    )
    old.steps[0].status = StepStatus.DONE
    old.steps[1].status = StepStatus.DONE
    fresh = build_plan("t", {"steps": [{"description": "x"}, {"description": "y"}]})
    plan = merge_revision(old, old.steps[:2], fresh, revision=2)

    assert [s.description for s in plan.steps] == ["a", "b", "x", "y"]
    assert [s.index for s in plan.steps] == [1, 2, 3, 4]
    assert all(s.plan_id == fresh.id for s in plan.steps)
    assert plan.steps[2].depends_on == [old.steps[1].id]
    assert plan.steps[3].depends_on == [fresh.steps[0].id]
    assert plan.revision == 2


def test_the_revised_plan_may_not_exceed_the_step_limit() -> None:
    too_many = {"steps": [{"description": f"s{i}"} for i in range(MAX_STEPS + 1)]}
    with pytest.raises(Exception, match="maximum"):
        build_plan("t", too_many)
