"""The approval flow (design doc section 5 and 6.2): pause, human decision, carry on."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fakes import ScriptedModel, call_tools, plan_json, say

from aegis.controller import Controller, InvalidTransition
from aegis.memory import EventLog
from aegis.model_client import Message
from aegis.models import EventType, Task, TaskState
from aegis.tools import default_registry

WRITE_A = ("fs_write", {"path": "a.txt", "content": "alpha\n"})
WRITE_B = ("fs_write", {"path": "b.txt", "content": "beta\n"})


@pytest.fixture
def events(tmp_path: Path) -> EventLog:
    return EventLog(tmp_path / "aegis.db")


def make(tmp_path: Path, events: EventLog, model: ScriptedModel) -> Controller:
    return Controller(model, default_registry(), events, workspaces_dir=tmp_path / "workspaces")


def run(controller: Controller, task_id: str) -> Task:
    return asyncio.run(controller.run(task_id))


def approve(controller: Controller, task_id: str) -> Task:
    return asyncio.run(controller.approve(task_id))


def types(events: EventLog, task_id: str, kind: EventType) -> list[dict[str, Any]]:
    return [e.payload for e in events.events(task_id, kind)]


def assert_p5(events: EventLog, task_id: str) -> None:
    assert events.unmatched_tool_calls(task_id) == []


# ---------------------------------------------------------------- pausing


def test_an_undeclared_write_pauses_the_task_before_anything_is_written(
    tmp_path: Path, events: EventLog
) -> None:
    model = ScriptedModel(say(plan_json("Write a.txt")), call_tools(WRITE_A))
    controller = make(tmp_path, events, model)
    task = controller.create_task("write a.txt")

    run(controller, task.id)

    assert task.state is TaskState.WAITING_FOR_APPROVAL
    assert not (task.workspace_path / "a.txt").exists()
    requested = types(events, task.id, EventType.APPROVAL_REQUESTED)
    assert len(requested) == 1
    assert requested[0]["tool_name"] == "fs_write"
    assert requested[0]["arguments"]["path"] == "a.txt"
    results = types(events, task.id, EventType.TOOL_RESULT)
    assert results[-1]["outcome"] == "approval_required"
    assert_p5(events, task.id)
    snapshot = controller.snapshot(task.id)
    assert snapshot["pending_approval"]["tool_name"] == "fs_write"
    assert snapshot["task"]["state"] == "waiting_for_approval"
    assert snapshot["plan"]["steps"][0]["status"] == "in_progress"


def test_a_waiting_task_cannot_be_run_again_only_approved(tmp_path: Path, events: EventLog) -> None:
    controller = make(tmp_path, events, ScriptedModel(say(plan_json("w")), call_tools(WRITE_A)))
    task = controller.create_task("write a.txt")
    run(controller, task.id)
    with pytest.raises(InvalidTransition, match="use approve"):
        run(controller, task.id)


# ---------------------------------------------------------------- approving


def test_approving_runs_the_call_and_the_task_completes(tmp_path: Path, events: EventLog) -> None:
    model = ScriptedModel(
        say(plan_json("Write a.txt")),
        call_tools(WRITE_A),
        say("Wrote a.txt"),
    )
    controller = make(tmp_path, events, model)
    task = controller.create_task("write a.txt", acceptance=["file_written:a.txt"])
    run(controller, task.id)

    finished = approve(controller, task.id)

    assert finished.state is TaskState.COMPLETED
    assert (task.workspace_path / "a.txt").read_text(encoding="utf-8") == "alpha\n"
    granted = types(events, task.id, EventType.APPROVAL_GRANTED)
    assert len(granted) == 1
    blocked_id = types(events, task.id, EventType.APPROVAL_REQUESTED)[0]["tool_call_id"]
    assert granted[0]["tool_call_id"] == blocked_id
    calls = types(events, task.id, EventType.TOOL_CALL)
    assert [c["approved"] for c in calls] == [False, True]  # blocked, then really run
    assert_p5(events, task.id)
    transitions = [
        f"{e['from']}->{e['to']}"
        for e in types(events, task.id, EventType.STATE_CHANGED)
        if e["scope"] == "task"
    ]
    assert transitions == [
        "created->running",
        "running->waiting_for_approval",
        "waiting_for_approval->running",
        "running->verifying",
        "verifying->completed",
    ]
    # Usage still adds up from events: the blocked call and the real call both count.
    rebuilt = events.recompute_usage(task.id)
    assert rebuilt.tool_calls == finished.usage.tool_calls == 2
    assert rebuilt.model_calls == finished.usage.model_calls


def test_a_new_process_can_approve_from_the_checkpoint_alone(
    tmp_path: Path, events: EventLog
) -> None:
    first = make(tmp_path, events, ScriptedModel(say(plan_json("w")), call_tools(WRITE_A)))
    task = first.create_task("write a.txt", acceptance=["file_written:a.txt"])
    run(first, task.id)

    seen: list[list[Message]] = []

    def finish(messages: list[Message]) -> Any:
        seen.append(messages)
        return say("done")

    # A brand new controller and model, as `aegis approve` in a fresh process would have.
    second = make(tmp_path, EventLog(tmp_path / "aegis.db"), ScriptedModel(finish))
    finished = approve(second, task.id)

    assert finished.state is TaskState.COMPLETED
    assert (task.workspace_path / "a.txt").exists()
    # The model was never asked for a new plan, and it was shown the result of its write.
    assert len(seen) == 1
    roles = [m.role for m in seen[0]]
    assert roles[-2:] == ["assistant", "tool"]
    assert "a.txt" in (seen[0][-1].content or "")


def test_calls_queued_behind_a_blocked_call_run_after_approval_in_order(
    tmp_path: Path, events: EventLog
) -> None:
    (tmp_path / "n.txt").write_text("note", encoding="utf-8")
    seen: list[list[Message]] = []

    def finish(messages: list[Message]) -> Any:
        seen.append(messages)
        return say("done")

    model = ScriptedModel(
        say(plan_json("Do three things")),
        call_tools(
            ("fs_read", {"path": "n.txt"}),
            WRITE_A,
            ("calculator", {"expression": "2 + 2"}),
        ),
        finish,
    )
    controller = make(tmp_path, events, model)
    task = controller.create_task("three things", attach=[tmp_path / "n.txt"])
    run(controller, task.id)
    assert task.state is TaskState.WAITING_FOR_APPROVAL
    ran_before = [c["tool_name"] for c in types(events, task.id, EventType.TOOL_CALL)]
    assert ran_before == ["fs_read", "fs_write"]  # the calculator has not run yet

    finished = approve(controller, task.id)

    assert finished.state is TaskState.COMPLETED
    ran = [c["tool_name"] for c in types(events, task.id, EventType.TOOL_CALL)]
    assert ran == ["fs_read", "fs_write", "fs_write", "calculator"]
    tool_messages = [m for m in seen[0] if m.role == "tool"]
    assert len(tool_messages) == 3  # one answer for each call the model made
    assert "4" in (tool_messages[2].content or "")
    assert_p5(events, task.id)


def test_two_blocked_calls_need_two_approvals(tmp_path: Path, events: EventLog) -> None:
    model = ScriptedModel(
        say(plan_json("Write two files")), call_tools(WRITE_A, WRITE_B), say("done")
    )
    controller = make(tmp_path, events, model)
    task = controller.create_task("two files")
    run(controller, task.id)
    assert not (task.workspace_path / "a.txt").exists()

    approve(controller, task.id)
    assert task.state is TaskState.WAITING_FOR_APPROVAL
    assert (task.workspace_path / "a.txt").exists()
    assert not (task.workspace_path / "b.txt").exists()  # the second write still waits

    approve(controller, task.id)
    assert task.state is TaskState.COMPLETED
    assert (task.workspace_path / "b.txt").exists()
    assert len(types(events, task.id, EventType.APPROVAL_REQUESTED)) == 2
    assert len(types(events, task.id, EventType.APPROVAL_GRANTED)) == 2


def test_waiting_time_does_not_count_against_the_wall_clock_budget(
    tmp_path: Path, events: EventLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [0.0]

    def ticking() -> float:
        now[0] += 1.0  # every look at the clock costs one second, so work takes visible time
        return now[0]

    monkeypatch.setattr("aegis.controller.time", SimpleNamespace(monotonic=ticking))
    controller = make(
        tmp_path, events, ScriptedModel(say(plan_json("w")), call_tools(WRITE_A), say("ok"))
    )
    task = controller.create_task("write a.txt")
    run(controller, task.id)
    worked_before = task.usage.wall_clock_s
    assert worked_before > 0

    now[0] += 10_000  # the human takes a very long time to answer
    approve(controller, task.id)

    assert task.usage.wall_clock_s >= worked_before  # earlier work is not forgotten
    assert task.usage.wall_clock_s < worked_before + 100  # the waiting is not counted


# ---------------------------------------------------------------- denying


def test_cancelling_a_waiting_task_denies_the_call_for_good(
    tmp_path: Path, events: EventLog
) -> None:
    controller = make(tmp_path, events, ScriptedModel(say(plan_json("w")), call_tools(WRITE_A)))
    task = controller.create_task("write a.txt")
    run(controller, task.id)

    controller.cancel(task.id)

    assert task.state is TaskState.CANCELLED
    assert not (task.workspace_path / "a.txt").exists()
    assert controller.snapshot(task.id)["pending_approval"] is None
    with pytest.raises(InvalidTransition):
        approve(controller, task.id)
    assert events.events(task.id)[-1].payload["to"] == "cancelled"


def test_approving_when_nothing_is_pending_is_an_error(tmp_path: Path, events: EventLog) -> None:
    controller = make(tmp_path, events, ScriptedModel())
    task = controller.create_task("anything")
    with pytest.raises(InvalidTransition, match="no approval is pending"):
        approve(controller, task.id)


# ---------------------------------------------------------------- declared scope


def test_a_declared_workspace_scope_needs_no_approval_for_writes(
    tmp_path: Path, events: EventLog
) -> None:
    model = ScriptedModel(say(plan_json("w")), call_tools(WRITE_A), say("ok"))
    controller = make(tmp_path, events, model)
    task = controller.create_task("write a.txt", allowed_effects=["workspace"])
    finished = run(controller, task.id)
    assert finished.state is TaskState.COMPLETED
    assert types(events, task.id, EventType.APPROVAL_REQUESTED) == []


def test_declaring_workspace_does_not_allow_running_code(tmp_path: Path, events: EventLog) -> None:
    model = ScriptedModel(say(plan_json("t")), call_tools(("shell_run_tests", {})))
    controller = make(tmp_path, events, model)
    task = controller.create_task("run tests", allowed_effects=["workspace"])
    run(controller, task.id)
    assert task.state is TaskState.WAITING_FOR_APPROVAL
    assert types(events, task.id, EventType.APPROVAL_REQUESTED)[0]["tool_name"] == "shell_run_tests"


def test_the_declared_scope_survives_a_restart(tmp_path: Path, events: EventLog) -> None:
    controller = make(tmp_path, events, ScriptedModel())
    task = controller.create_task("x", allowed_effects=["workspace", "execute"])
    reloaded = make(tmp_path, EventLog(tmp_path / "aegis.db"), ScriptedModel())
    assert reloaded.get_task(task.id).allowed_effects == ["execute", "workspace"]


@pytest.mark.parametrize("bad", [["external"], ["workspace", "root"], ["none"]])
def test_only_workspace_and_execute_can_be_declared(
    tmp_path: Path, events: EventLog, bad: list[str]
) -> None:
    controller = make(tmp_path, events, ScriptedModel())
    with pytest.raises(ValueError, match="cannot pre-approve"):
        controller.create_task("x", allowed_effects=bad)


# ---------------------------------------------------------------- tests, edits and stale evidence

BUGGY = ("fs_write", {"path": "calc.py", "content": "def add(a, b):\n    return a - b\n"})
TESTS = (
    "fs_write",
    {
        "path": "test_calc.py",
        "content": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    },
)
RUN_TESTS = ("shell_run_tests", {})
FIX = ("fs_edit", {"path": "calc.py", "old": "a - b", "new": "a + b"})
BOTH = ["workspace", "execute"]


def test_fixing_a_bug_then_passing_tests_completes_the_task(
    tmp_path: Path, events: EventLog
) -> None:
    model = ScriptedModel(
        say(plan_json("Create the project and fix it")),
        call_tools(BUGGY, TESTS),
        call_tools(RUN_TESTS),
        call_tools(FIX),
        call_tools(RUN_TESTS),
        say("Fixed add(); tests pass."),
    )
    controller = make(tmp_path, events, model)
    task = controller.create_task(
        "make add() work",
        acceptance=["tests_pass", "file_written:calc.py"],
        allowed_effects=BOTH,
    )
    finished = run(controller, task.id)
    assert finished.state is TaskState.COMPLETED, finished.last_error
    edit = next(
        e for e in types(events, task.id, EventType.TOOL_RESULT) if e["tool_name"] == "fs_edit"
    )
    assert edit["result"]["diff"]  # a non-empty diff is on the record


def test_tests_that_passed_before_a_later_edit_do_not_count(
    tmp_path: Path, events: EventLog
) -> None:
    good = ("fs_write", {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"})
    tweak = ("fs_edit", {"path": "calc.py", "old": "a + b", "new": "b + a"})
    model = ScriptedModel(
        say(plan_json("Write, test, then edit")),
        call_tools(good, TESTS),
        call_tools(RUN_TESTS),
        call_tools(tweak),  # a change AFTER the passing run
        say("All done, tests pass."),
    )
    controller = make(tmp_path, events, model)
    task = controller.create_task("add", acceptance=["tests_pass"], allowed_effects=BOTH)
    finished = run(controller, task.id)
    assert finished.state is TaskState.FAILED
    assert "files changed afterwards" in (finished.last_error or "")


def test_failing_tests_do_not_count_as_passing(tmp_path: Path, events: EventLog) -> None:
    model = ScriptedModel(
        say(plan_json("Write and test")),
        call_tools(BUGGY, TESTS),
        call_tools(RUN_TESTS),
        say("tests pass"),  # a false claim
    )
    controller = make(tmp_path, events, model)
    task = controller.create_task("add", acceptance=["tests_pass"], allowed_effects=BOTH)
    finished = run(controller, task.id)
    assert finished.state is TaskState.FAILED
    assert "no test run with exit code 0" in (finished.last_error or "")


def test_a_dry_run_is_not_a_file_change(tmp_path: Path, events: EventLog) -> None:
    good = ("fs_write", {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"})
    preview = (
        "fs_edit",
        {"path": "calc.py", "old": "a + b", "new": "b + a", "dry_run": True},
    )
    model = ScriptedModel(
        say(plan_json("Write, test, preview")),
        call_tools(good, TESTS),
        call_tools(RUN_TESTS),
        call_tools(preview),
        say("ok"),
    )
    controller = make(tmp_path, events, model)
    task = controller.create_task("add", acceptance=["tests_pass"], allowed_effects=BOTH)
    assert run(controller, task.id).state is TaskState.COMPLETED
