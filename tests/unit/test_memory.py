import sqlite3
from pathlib import Path
from typing import Any

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


def test_checkpoints_are_saved_replaced_and_loaded() -> None:
    log = EventLog()
    assert log.load_checkpoint("t1") is None
    log.save_checkpoint("t1", {"state": "created"})
    log.save_checkpoint("t1", {"state": "running"})  # replaces: checkpoints are mutable
    log.save_checkpoint("t2", {"state": "other"})
    assert log.load_checkpoint("t1") == {"state": "running"}
    assert log.load_checkpoint("t2") == {"state": "other"}


def test_usage_is_recomputed_from_events_alone() -> None:
    log = EventLog()
    log.append(
        "t1",
        EventType.MODEL_CALL,
        {"attempts": 1, "input_tokens": 60, "output_tokens": 40, "dollars": "0.10"},
    )
    log.append(
        "t1",
        EventType.MODEL_CALL,
        {"attempts": 3, "input_tokens": 0, "output_tokens": 0, "dollars": "0"},  # failed retries
    )
    log.append("t1", EventType.TOOL_CALL, {}, correlation_id="a")
    log.append("t1", EventType.TOOL_RESULT, {}, correlation_id="a")
    log.append("t1", EventType.MODEL_CALL, {"attempts": 1})  # missing fields default to zero
    log.append("t2", EventType.MODEL_CALL, {"attempts": 9, "input_tokens": 999})  # other task
    usage = log.recompute_usage("t1")
    assert usage.model_calls == 5
    assert usage.tool_calls == 1
    assert usage.tokens_total == 100
    assert str(usage.dollars) == "0.10"


def test_opening_retries_a_transient_windows_disk_io_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI on Windows: reopening the log right after a kill raised 'disk I/O error'."""
    import sqlite3

    real_connect = sqlite3.connect
    attempts: list[int] = []

    class Flaky:
        def __init__(self, path: str) -> None:
            self._conn = real_connect(path)
            attempts.append(1)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._conn, name)

        def __setattr__(self, name: str, value: Any) -> None:
            if name == "_conn":
                object.__setattr__(self, name, value)
            else:
                setattr(self._conn, name, value)

        def execute(self, sql: str, *args: Any) -> Any:
            if "journal_mode" in sql and len(attempts) < 3:
                raise sqlite3.OperationalError("disk I/O error")
            return self._conn.execute(sql, *args)

    monkeypatch.setattr("aegis.memory.sqlite3.connect", Flaky)
    monkeypatch.setattr("aegis.memory.OPEN_RETRY_DELAY_S", 0)
    log = EventLog(tmp_path / "a.db")
    assert len(attempts) == 3 and log.schema_version() > 0


def test_a_persistent_disk_error_is_raised_after_the_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    calls: list[int] = []

    def always_fails(path: str) -> Any:
        calls.append(1)
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr("aegis.memory.sqlite3.connect", always_fails)
    monkeypatch.setattr("aegis.memory.OPEN_RETRY_DELAY_S", 0)
    with pytest.raises(sqlite3.OperationalError):
        EventLog(tmp_path / "a.db")
    assert len(calls) == 6


def test_other_database_errors_are_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    calls: list[int] = []

    def broken(path: str) -> Any:
        calls.append(1)
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr("aegis.memory.sqlite3.connect", broken)
    with pytest.raises(sqlite3.OperationalError):
        EventLog(tmp_path / "a.db")
    assert calls == [1]
