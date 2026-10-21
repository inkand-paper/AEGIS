"""Crash recovery (design doc 8.2). A "crash" here is an exception that is not an `Exception`,
so nothing in the controller catches it: the task is left exactly as a killed process would
leave it, apart from the lease, which a dead process cannot release (tests that care set it)."""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fakes import ScriptedModel, call_tools, plan_json, say
from pydantic import BaseModel, ConfigDict

from aegis.controller import Controller, InvalidTransition
from aegis.memory import EventLog, LeaseHeld
from aegis.model_client import Message
from aegis.models import EventType, Task, TaskState
from aegis.tools import default_registry
from aegis.tools.base import Tool, ToolContext, ToolRegistry


class SimulatedCrash(BaseException):
    """Stands in for SIGKILL or a power cut: nothing is allowed to handle it."""


def boom(messages: list[Message]) -> Any:
    raise SimulatedCrash


class LineArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line: str


class Done(BaseModel):
    note: str


class LedgerTool(Tool[LineArgs, Done]):
    """Appends a line to ledger.txt. NOT idempotent: running it twice leaves two lines."""

    name = "ledger_append"
    description = "Append a line to the ledger."
    arg_model = LineArgs
    result_model = Done
    side_effects = "workspace"
    requires_network = False
    idempotent = False

    def __init__(self, crash: str | None = None) -> None:
        self.crash = crash  # "before" or "after" the effect, on the first run only

    async def execute(self, args: LineArgs, ctx: ToolContext) -> Done:
        crash, self.crash = self.crash, None
        if crash == "before":
            raise SimulatedCrash
        with (ctx.workspace / "ledger.txt").open("a", encoding="utf-8") as handle:
            handle.write(args.line + "\n")
        if crash == "after":
            raise SimulatedCrash
        return Done(note=f"appended {args.line}")


class LookupTool(Tool[LineArgs, Done]):
    """Read-only and idempotent: safe to run again after a crash."""

    name = "lookup"
    description = "Look something up."
    arg_model = LineArgs
    result_model = Done
    side_effects = "none"
    requires_network = False
    idempotent = True

    def __init__(self, crash: bool = False) -> None:
        self.crash = crash
        self.runs = 0

    async def execute(self, args: LineArgs, ctx: ToolContext) -> Done:
        self.runs += 1
        if self.crash:
            self.crash = False
            raise SimulatedCrash
        return Done(note=f"found {args.line}")


def registry(*extra: Tool[Any, Any]) -> ToolRegistry:
    reg = default_registry()
    for tool in extra:
        reg.register(tool)
    return reg


def process(
    tmp_path: Path, model: ScriptedModel, *extra: Tool[Any, Any], ttl: float = 30.0
) -> Controller:
    """A fresh 'process': its own database connection, controller and tool instances."""
    return Controller(
        model,
        registry(*extra),
        EventLog(tmp_path / "aegis.db"),
        workspaces_dir=tmp_path / "workspaces",
        lease_ttl_s=ttl,
    )


def go(controller: Controller, task_id: str) -> Task:
    return asyncio.run(controller.run(task_id))


def crash_during_run(controller: Controller, task_id: str) -> None:
    with pytest.raises(SimulatedCrash):
        go(controller, task_id)


def resume(controller: Controller, task_id: str) -> Task:
    return asyncio.run(controller.resume(task_id))


def events_of(tmp_path: Path, task_id: str, kind: EventType) -> list[dict[str, Any]]:
    return [e.payload for e in EventLog(tmp_path / "aegis.db").events(task_id, kind)]


def ledger(tmp_path: Path, task_id: str) -> list[str]:
    path = tmp_path / "workspaces" / task_id / "ledger.txt"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def new_task(tmp_path: Path, model: ScriptedModel, *extra: Tool[Any, Any], **kwargs: Any) -> str:
    controller = process(tmp_path, model, *extra)
    task = controller.create_task("do the work", **kwargs)
    crash_during_run(controller, task.id)
    return task.id


# ---------------------------------------------------------------- where the crash happens


def test_a_crash_in_the_middle_of_a_step_resumes_without_replanning(tmp_path: Path) -> None:
    first = ScriptedModel(
        say(plan_json("Calculate", "Report")),
        call_tools(("calculator", {"expression": "17 * 23"})),
        boom,  # the process dies while waiting for the model
    )
    task_id = new_task(tmp_path, first, acceptance=["result_contains:391"])

    seen: list[list[Message]] = []

    def step_one_done(messages: list[Message]) -> Any:
        seen.append(messages)
        return say("17 * 23 = 391")

    second = ScriptedModel(step_one_done, say("The answer is 391."))
    task = resume(process(tmp_path, second), task_id)

    assert task.state is TaskState.COMPLETED, task.last_error
    assert len(events_of(tmp_path, task_id, EventType.PLAN_CREATED)) == 1  # no new plan
    # The model was shown what already happened in this step, and told it was interrupted.
    context = seen[0]
    assert "interrupted and is being resumed" in (context[1].content or "")
    assert [m.role for m in context[2:]] == ["assistant", "tool"]
    assert "391" in (context[-1].content or "")
    resumed = events_of(tmp_path, task_id, EventType.TASK_RESUMED)
    assert len(resumed) == 1 and resumed[0]["resumed_step"] == 1
    snapshot = EventLog(tmp_path / "aegis.db").load_checkpoint(task_id)
    assert snapshot is not None and snapshot["task"]["state"] == "completed"


def test_finished_steps_are_not_run_again_after_a_crash_between_steps(tmp_path: Path) -> None:
    first = ScriptedModel(
        say(plan_json("Calculate", "Report")),
        call_tools(("calculator", {"expression": "6 * 7"})),
        say("6 * 7 = 42"),  # step 1 finishes
        boom,  # step 2 starts and the process dies
    )
    task_id = new_task(tmp_path, first)

    second = ScriptedModel(say("Final answer: 42."))  # only step 2 is left, nothing else is asked
    task = resume(process(tmp_path, second), task_id)

    assert task.state is TaskState.COMPLETED, task.last_error
    assert second.calls and len(second.calls) == 1
    prompt = second.calls[0][1].content or ""
    assert "6 * 7 = 42" in prompt  # step 1's result was restored from the events


def test_a_crash_before_the_plan_exists_just_plans_again(tmp_path: Path) -> None:
    task_id = new_task(tmp_path, ScriptedModel(boom))
    second = ScriptedModel(
        say(plan_json("Calculate")),
        call_tools(("calculator", {"expression": "2 + 2"})),
        say("4"),
    )
    task = resume(process(tmp_path, second), task_id)
    assert task.state is TaskState.COMPLETED, task.last_error


def test_a_crash_while_verifying_verifies_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = process(
        tmp_path,
        ScriptedModel(
            say(plan_json("Calculate")),
            call_tools(("calculator", {"expression": "2 + 2"})),
            say("4"),
        ),
    )
    task = controller.create_task("add", acceptance=["result_contains:4"])

    def crash_in_verifier(**kwargs: Any) -> Any:
        raise SimulatedCrash

    monkeypatch.setattr(controller._verifier, "verify", crash_in_verifier)
    crash_during_run(controller, task.id)
    assert EventLog(tmp_path / "aegis.db").load_checkpoint(task.id)["task"]["state"] == "verifying"  # type: ignore[index]

    finished = resume(process(tmp_path, ScriptedModel()), task.id)

    assert finished.state is TaskState.COMPLETED
    assert finished.verification_result_id is not None


def test_a_checkpoint_that_is_behind_the_events_is_corrected(tmp_path: Path) -> None:
    """The checkpoint is written after the event. Crash in between, and it is one step behind."""
    first = ScriptedModel(
        say(plan_json("Calculate", "Report")),
        call_tools(("calculator", {"expression": "6 * 7"})),
        say("6 * 7 = 42"),
        boom,
    )
    controller = process(tmp_path, first)
    task = controller.create_task("do it")
    crash_during_run(controller, task.id)
    log = EventLog(tmp_path / "aegis.db")
    snapshot = log.load_checkpoint(task.id)
    assert snapshot is not None
    for step in snapshot["plan"]["steps"]:  # roll the checkpoint back: nothing looks done
        step["status"] = "pending"
        step["attempts"] = 0
    snapshot["step_results"] = {}
    log.save_checkpoint(task.id, snapshot)

    second = ScriptedModel(say("Final answer: 42."))
    finished = resume(process(tmp_path, second), task.id)

    assert finished.state is TaskState.COMPLETED, finished.last_error
    assert len(second.calls) == 1  # step 1 was recognised as done from its event
    assert "6 * 7 = 42" in (second.calls[0][1].content or "")  # and its result was recovered


# ---------------------------------------------------------------- calls that were cut off


def test_an_interrupted_idempotent_call_is_run_again(tmp_path: Path) -> None:
    crashing = LookupTool(crash=True)
    first = ScriptedModel(say(plan_json("Look it up")), call_tools(("lookup", {"line": "milk"})))
    task_id = new_task(tmp_path, first, crashing)
    log = EventLog(tmp_path / "aegis.db")
    assert len(log.unmatched_tool_calls(task_id)) == 1  # the call has no result: indeterminate

    again = LookupTool()
    second = ScriptedModel(say("Found it: milk."))
    task = resume(process(tmp_path, second, again), task_id)

    assert task.state is TaskState.COMPLETED, task.last_error
    assert again.runs == 1  # it was executed again, by the controller, without asking the model
    results = events_of(tmp_path, task_id, EventType.TOOL_RESULT)
    assert [r["outcome"] for r in results] == ["interrupted", "ok"]
    assert log.unmatched_tool_calls(task_id) == []  # every call has a result again (P5)
    assert [m.role for m in second.calls[0][2:]] == [
        "assistant",
        "tool",
    ]  # the model saw the result


def test_an_interrupted_non_idempotent_call_is_never_repeated_blindly(tmp_path: Path) -> None:
    """Phase 4 exit criterion: resume reaches WAITING_FOR_APPROVAL, never a silent duplicate."""
    first = ScriptedModel(
        say(plan_json("Record it")), call_tools(("ledger_append", {"line": "paid 100"}))
    )
    task_id = new_task(tmp_path, first, LedgerTool(crash="after"), allowed_effects=["workspace"])
    assert ledger(tmp_path, task_id) == ["paid 100"]  # the effect happened, the result did not

    never_asked = ScriptedModel()
    task = resume(process(tmp_path, never_asked, LedgerTool()), task_id)

    assert task.state is TaskState.WAITING_FOR_APPROVAL
    assert ledger(tmp_path, task_id) == ["paid 100"]  # still exactly one line
    assert never_asked.calls == []
    requested = events_of(tmp_path, task_id, EventType.APPROVAL_REQUESTED)
    assert requested[-1]["tool_name"] == "ledger_append"
    assert "may have partly run" in requested[-1]["reason"]
    snapshot = EventLog(tmp_path / "aegis.db").load_checkpoint(task_id)
    assert snapshot is not None
    assert snapshot["recovery"]["needs_human"] is True
    assert snapshot["pending_approval"]["tool_name"] == "ledger_append"


def test_a_human_can_approve_the_repeat_or_cancel_it(tmp_path: Path) -> None:
    def waiting_task(name: str) -> tuple[Controller, str]:
        sub = tmp_path / name
        first = ScriptedModel(
            say(plan_json("Record it")), call_tools(("ledger_append", {"line": "paid 100"}))
        )
        task_id = new_task(sub, first, LedgerTool(crash="after"), allowed_effects=["workspace"])
        controller = process(sub, ScriptedModel(say("Recorded.")), LedgerTool())
        assert resume(controller, task_id).state is TaskState.WAITING_FOR_APPROVAL
        return controller, task_id

    controller, task_id = waiting_task("approve")
    done = asyncio.run(controller.approve(task_id))
    assert done.state is TaskState.COMPLETED, done.last_error
    assert ledger(tmp_path / "approve", task_id) == ["paid 100", "paid 100"]  # the human chose this

    controller, task_id = waiting_task("cancel")
    controller.cancel(task_id)
    assert controller.get_task(task_id).state is TaskState.CANCELLED
    assert ledger(tmp_path / "cancel", task_id) == ["paid 100"]  # nothing more was written


def test_a_non_idempotent_call_cut_off_before_its_effect_also_waits(tmp_path: Path) -> None:
    """We cannot tell 'before' from 'after', so both are treated the same: a human decides."""
    first = ScriptedModel(say(plan_json("Record it")), call_tools(("ledger_append", {"line": "x"})))
    task_id = new_task(tmp_path, first, LedgerTool(crash="before"), allowed_effects=["workspace"])
    assert ledger(tmp_path, task_id) == []
    task = resume(process(tmp_path, ScriptedModel(), LedgerTool()), task_id)
    assert task.state is TaskState.WAITING_FOR_APPROVAL


def test_a_repeated_change_after_resume_is_suppressed(tmp_path: Path) -> None:
    """The change succeeded and was recorded; the process died before the model moved on."""
    write = ("ledger_append", {"line": "paid 100"})
    first = ScriptedModel(say(plan_json("Record it")), call_tools(write), boom)
    task_id = new_task(tmp_path, first, LedgerTool(), allowed_effects=["workspace"])
    assert ledger(tmp_path, task_id) == ["paid 100"]

    forgetful = ScriptedModel(call_tools(write), say("Recorded."))  # proposes it again
    task = resume(process(tmp_path, forgetful, LedgerTool()), task_id)

    assert task.state is TaskState.COMPLETED, task.last_error
    assert ledger(tmp_path, task_id) == ["paid 100"]  # not written twice
    results = events_of(tmp_path, task_id, EventType.TOOL_RESULT)
    assert [r["deduplicated"] for r in results] == [False, True]


# ---------------------------------------------------------------- usage drift


def test_corrupted_usage_is_recomputed_from_events_and_reported(tmp_path: Path) -> None:
    """Phase 4 exit criterion: corrupt `usage` in the DB, resume recomputes it and detects drift."""
    first = ScriptedModel(
        say(plan_json("Calculate")),
        call_tools(("calculator", {"expression": "1 + 1"})),
        boom,
    )
    task_id = new_task(tmp_path, first)
    truth = EventLog(tmp_path / "aegis.db").recompute_usage(task_id)
    assert truth.model_calls == 2 and truth.tool_calls == 1  # the cut-off call is not counted

    with sqlite3.connect(tmp_path / "aegis.db") as conn:  # tamper with the stored state
        raw = conn.execute("SELECT snapshot FROM checkpoints WHERE task_id = ?", (task_id,))
        snapshot = json.loads(raw.fetchone()[0])
        snapshot["task"]["usage"].update(model_calls=1, tool_calls=99, tokens_total=7)
        conn.execute(
            "UPDATE checkpoints SET snapshot = ? WHERE task_id = ?",
            (json.dumps(snapshot), task_id),
        )

    second = ScriptedModel(say("2"))
    task = resume(process(tmp_path, second), task_id)

    assert task.state is TaskState.COMPLETED, task.last_error
    drift = events_of(tmp_path, task_id, EventType.TASK_RESUMED)[0]["usage_drift"]
    assert set(drift) == {"model_calls", "tool_calls", "tokens_total"}
    assert drift["model_calls"] == {"recorded": "1", "from_events": "2"}
    assert drift["tool_calls"]["recorded"] == "99"
    rebuilt = EventLog(tmp_path / "aegis.db").recompute_usage(task_id)
    assert task.usage.tool_calls == rebuilt.tool_calls == 1  # the corrupt number is gone
    assert task.usage.model_calls == rebuilt.model_calls == 3  # two before + one after resume


def test_a_clean_resume_reports_no_drift(tmp_path: Path) -> None:
    first = ScriptedModel(say(plan_json("Calculate")), boom)
    task_id = new_task(tmp_path, first)
    resume(
        process(
            tmp_path, ScriptedModel(call_tools(("calculator", {"expression": "1+1"})), say("2"))
        ),
        task_id,
    )
    assert events_of(tmp_path, task_id, EventType.TASK_RESUMED)[0]["usage_drift"] == {}


# ---------------------------------------------------------------- who may resume, and when


def test_finished_tasks_cannot_be_resumed(tmp_path: Path) -> None:
    controller = process(
        tmp_path,
        ScriptedModel(
            say(plan_json("Calculate")),
            call_tools(("calculator", {"expression": "1 + 1"})),
            say("2"),
        ),
    )
    task = controller.create_task("add")
    go(controller, task.id)
    with pytest.raises(InvalidTransition, match="already completed"):
        resume(process(tmp_path, ScriptedModel()), task.id)


def test_a_task_waiting_for_approval_is_approved_not_resumed(tmp_path: Path) -> None:
    controller = process(
        tmp_path,
        ScriptedModel(
            say(plan_json("Write")), call_tools(("fs_write", {"path": "a", "content": "x"}))
        ),
    )
    task = controller.create_task("write")
    go(controller, task.id)
    with pytest.raises(InvalidTransition, match="use approve or cancel"):
        resume(process(tmp_path, ScriptedModel()), task.id)


def test_resuming_a_task_that_never_started_simply_runs_it(tmp_path: Path) -> None:
    controller = process(tmp_path, ScriptedModel())
    task = controller.create_task("add")
    second = process(
        tmp_path,
        ScriptedModel(
            say(plan_json("Calculate")),
            call_tools(("calculator", {"expression": "1 + 1"})),
            say("2"),
        ),
    )
    assert resume(second, task.id).state is TaskState.COMPLETED


def test_resume_of_an_unknown_task_is_reported(tmp_path: Path) -> None:
    from aegis.controller import TaskNotFound

    with pytest.raises(TaskNotFound):
        resume(process(tmp_path, ScriptedModel()), "nope")


# ---------------------------------------------------------------- one worker at a time


def test_a_task_being_worked_on_by_a_live_process_cannot_be_resumed(tmp_path: Path) -> None:
    task_id = new_task(tmp_path, ScriptedModel(boom))
    EventLog(tmp_path / "aegis.db").acquire_lease(task_id, "someone-else", ttl_s=300)
    before = len(EventLog(tmp_path / "aegis.db").events(task_id))

    with pytest.raises(LeaseHeld, match="someone-else"):
        resume(process(tmp_path, ScriptedModel()), task_id)

    assert len(EventLog(tmp_path / "aegis.db").events(task_id)) == before  # nothing was touched


def test_a_dead_process_lease_expires_and_the_task_can_be_resumed(tmp_path: Path) -> None:
    task_id = new_task(tmp_path, ScriptedModel(boom))
    log = EventLog(tmp_path / "aegis.db")
    log.acquire_lease(task_id, "dead-process", ttl_s=1, now=1.0)  # last heard of long ago
    second = ScriptedModel(
        say(plan_json("Calculate")),
        call_tools(("calculator", {"expression": "1 + 1"})),
        say("2"),
    )
    assert resume(process(tmp_path, second, ttl=30), task_id).state is TaskState.COMPLETED


def test_the_lease_is_released_when_the_work_stops(tmp_path: Path) -> None:
    controller = process(tmp_path, ScriptedModel(boom))
    task = controller.create_task("x")
    crash_during_run(controller, task.id)
    # The crashed worker released its lease on the way out, so an immediate resume is allowed.
    EventLog(tmp_path / "aegis.db").acquire_lease(task.id, "next", ttl_s=30)


def test_a_worker_that_loses_its_lease_stops_without_writing(tmp_path: Path) -> None:
    from aegis.controller import LeaseLost

    controller = process(tmp_path, ScriptedModel(say(plan_json("A")), say("never used")), ttl=0.3)
    task = controller.create_task("x")
    log = EventLog(tmp_path / "aegis.db")

    async def stolen() -> None:
        await asyncio.sleep(0.1)
        with sqlite3.connect(tmp_path / "aegis.db") as conn:  # another process takes the lease
            conn.execute("UPDATE leases SET owner = 'thief' WHERE task_id = ?", (task.id,))

    async def main() -> None:
        thief = asyncio.create_task(stolen())
        original = controller._model.generate

        async def slow(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0.5)  # a long model call, long enough to miss a heartbeat
            return await original(*args, **kwargs)

        controller._model.generate = slow  # type: ignore[method-assign]
        try:
            await controller.run(task.id)
        finally:
            await thief

    events_before = len(log.events(task.id))
    with pytest.raises(LeaseLost):
        asyncio.run(main())
    state = log.load_checkpoint(task.id)["task"]["state"]  # type: ignore[index]
    assert state == "running"  # we did not mark it failed or cancelled: the new owner decides
    assert len(log.events(task.id)) <= events_before + 3
