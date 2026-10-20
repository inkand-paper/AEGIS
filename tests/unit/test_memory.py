import sqlite3
from pathlib import Path

import pytest

from aegis.memory import EventLog
from aegis.models import EventType


def test_append_and_read_back_in_order() -> None:
    log = EventLog()
    first = log.append("t1", EventType.TASK_CREATED, {"objective": "demo"})
    second = log.append("t1", EventType.STATE_CHANGED, {"to": "running"})
    log.append("t2", EventType.TASK_CREATED, {})  # another task, must not leak in
    events = log.events("t1")
    assert [e.id for e in events] == [first.id, second.id]
    assert events[0].payload == {"objective": "demo"}
    assert [e.type for e in log.events("t1", EventType.STATE_CHANGED)] == [EventType.STATE_CHANGED]


def test_events_are_append_only() -> None:
    log = EventLog()
    event = log.append("t1", EventType.ERROR, {"message": "original"})
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        log._conn.execute("UPDATE events SET payload = '{}' WHERE id = ?", (event.id,))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        log._conn.execute("DELETE FROM events WHERE id = ?", (event.id,))
    assert log.events("t1")[0].payload == {"message": "original"}


def test_events_survive_closing_and_reopening(tmp_path: Path) -> None:
    path = tmp_path / "data" / "aegis.db"  # the data/ folder is created on demand
    log = EventLog(path)
    log.append("t1", EventType.TASK_CREATED, {"n": 1})
    log.close()
    reopened = EventLog(path)
    assert [e.payload for e in reopened.events("t1")] == [{"n": 1}]


def test_unmatched_tool_calls_finds_calls_without_results() -> None:
    log = EventLog()
    log.append("t1", EventType.TOOL_CALL, {}, correlation_id="a")
    log.append("t1", EventType.TOOL_RESULT, {}, correlation_id="a")
    log.append("t1", EventType.TOOL_CALL, {}, correlation_id="b")  # crashed before its result
    log.append("t2", EventType.TOOL_CALL, {}, correlation_id="c")  # different task
    assert [e.correlation_id for e in log.unmatched_tool_calls("t1")] == ["b"]