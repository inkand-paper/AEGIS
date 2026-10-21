import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fakes import ScriptedModel, call_tools, plan_json, say

from aegis.controller import Controller
from aegis.memory import EventLog
from aegis.models import EventType, TaskState
from aegis.replay import replay
from aegis.tools import default_registry


def boom(messages: Any) -> Any:
    raise KeyboardInterrupt  # not an Exception: leaves the task as a killed process would


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "aegis.db"


def controller(tmp_path: Path, db: Path, model: ScriptedModel) -> Controller:
    return Controller(
        model, default_registry(), EventLog(db), workspaces_dir=tmp_path / "workspaces"
    )


def finished_task(tmp_path: Path, db: Path) -> str:
    c = controller(
        tmp_path,
        db,
        ScriptedModel(
            say(plan_json("Calculate")),
            call_tools(("calculator", {"expression": "17 * 23"})),
            say("391"),
        ),
    )
    task = c.create_task("multiply", acceptance=["result_contains:391"])
    asyncio.run(c.run(task.id))
    assert task.state is TaskState.COMPLETED
    return task.id


def test_a_clean_finished_task_replays_ok(tmp_path: Path, db: Path) -> None:
    task_id = finished_task(tmp_path, db)
    report = replay(EventLog(db), task_id)
    assert report.ok, report.problems
    assert report.state is TaskState.COMPLETED
    assert report.event_count > 8
    assert report.usage.tool_calls == 1
    assert list(report.step_statuses.values()) == ["done"]


def test_a_task_waiting_for_approval_replays_ok(tmp_path: Path, db: Path) -> None:
    c = controller(
        tmp_path,
        db,
        ScriptedModel(
            say(plan_json("Write")), call_tools(("fs_write", {"path": "a", "content": "x"}))
        ),
    )
    task = c.create_task("write")
    asyncio.run(c.run(task.id))
    report = replay(EventLog(db), task.id)
    assert report.ok, report.problems
    assert report.state is TaskState.WAITING_FOR_APPROVAL


def test_a_crashed_task_is_ok_but_notes_what_resume_will_repair(tmp_path: Path, db: Path) -> None:
    c = controller(
        tmp_path,
        db,
        ScriptedModel(
            say(plan_json("Calculate")), call_tools(("calculator", {"expression": "1+1"})), boom
        ),
    )
    task = c.create_task("add")
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(c.run(task.id))
    report = replay(EventLog(db), task.id)
    assert report.ok, report.problems
    assert report.state is TaskState.RUNNING


def test_a_crash_inside_a_tool_call_leaves_a_note_not_a_problem(tmp_path: Path, db: Path) -> None:
    log = EventLog(db)
    task_id = finished_task(tmp_path, db)
    # Simulate the same kind of leftover a SIGKILL leaves: a call with no result, task still open.
    other = "open-task"
    log.append(other, EventType.TASK_CREATED, {"objective": "x"})
    log.append(
        other,
        EventType.STATE_CHANGED,
        {"scope": "task", "from": "created", "to": "running", "reason": "", "error": None},
    )
    log.append(other, EventType.TOOL_CALL, {"tool_call_id": "c1", "tool_name": "x"}, "c1")
    log.save_checkpoint(
        other,
        {
            "task": {
                "state": "running",
                "usage": {"model_calls": 0, "tool_calls": 1, "tokens_total": 0, "dollars": "0"},
            },
            "plan": None,
        },
    )
    report = replay(log, other)
    assert report.ok
    assert any("cut off" in note for note in report.notes)
    assert replay(log, task_id).ok


# ---------------------------------------------------------------- a log that breaks the rules


def append(
    db: Path, task_id: str, kind: EventType, payload: dict[str, Any], corr: str | None = None
) -> None:
    EventLog(db).append(task_id, kind, payload, corr)


def test_an_illegal_state_change_is_a_problem(tmp_path: Path, db: Path) -> None:
    task_id = finished_task(tmp_path, db)
    c = controller(tmp_path, db, ScriptedModel())
    task = c.create_task("other")
    append(
        db,
        task.id,
        EventType.STATE_CHANGED,
        {"scope": "task", "from": "created", "to": "completed", "reason": "", "error": None},
    )
    report = replay(EventLog(db), task.id)
    assert not report.ok
    assert any("illegal change created -> completed" in p for p in report.problems)
    assert any("COMPLETED without a passing verification" in p for p in report.problems)
    assert replay(EventLog(db), task_id).ok  # the other task is unaffected


def test_events_after_the_end_are_a_problem(tmp_path: Path, db: Path) -> None:
    task_id = finished_task(tmp_path, db)
    append(db, task_id, EventType.ERROR, {"note": "late"})
    report = replay(EventLog(db), task_id)
    assert any("came after the task ended" in p for p in report.problems)


def test_a_result_without_a_call_and_a_double_result_are_problems(tmp_path: Path, db: Path) -> None:
    task_id = finished_task(tmp_path, db)
    log = EventLog(db)
    existing = log.events(task_id, EventType.TOOL_RESULT)[0]
    log.append(task_id, EventType.TOOL_RESULT, {"success": True}, existing.correlation_id)
    log.append(task_id, EventType.TOOL_RESULT, {"success": True}, "ghost")
    problems = replay(log, task_id).problems
    assert any("second result" in p for p in problems)
    assert any("never recorded" in p for p in problems)


def test_evidence_from_a_failed_call_is_a_problem(tmp_path: Path, db: Path) -> None:
    task_id = finished_task(tmp_path, db)
    log = EventLog(db)
    log.append(task_id, EventType.EVIDENCE_ADDED, {"source_tool_call_id": "nope"}, "nope")
    assert any(
        "evidence from a call that did not succeed" in p for p in replay(log, task_id).problems
    )


def test_a_finished_task_whose_checkpoint_disagrees_is_a_problem(tmp_path: Path, db: Path) -> None:
    task_id = finished_task(tmp_path, db)
    with sqlite3.connect(db) as conn:
        snapshot = json.loads(conn.execute("SELECT snapshot FROM checkpoints").fetchone()[0])
        snapshot["task"]["usage"]["tool_calls"] = 42
        snapshot["plan"]["steps"][0]["status"] = "pending"
        conn.execute("UPDATE checkpoints SET snapshot = ?", (json.dumps(snapshot),))
    problems = replay(EventLog(db), task_id).problems
    assert any("usage.tool_calls: checkpoint 42, events 1" in p for p in problems)
    assert any("step 1: checkpoint pending, events done" in p for p in problems)


def test_an_unknown_task_does_not_replay(db: Path) -> None:
    report = replay(EventLog(db), "nope")
    assert not report.ok
    assert report.event_count == 0
