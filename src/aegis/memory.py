"""Append-only event log in SQLite (design doc sections 4.3, 5.5 and 8).

The log is the source of truth for "what actually happened". SQLite triggers make UPDATE and
DELETE on it fail, so even a bug in our own code cannot rewrite history.
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aegis.models import Event, EventType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id        TEXT NOT NULL,
    type           TEXT NOT NULL,
    payload        TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    correlation_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_task ON events (task_id, id);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
"""


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        id=row["id"],
        task_id=row["task_id"],
        type=EventType(row["type"]),
        payload=json.loads(row["payload"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        correlation_id=row["correlation_id"],
    )


class EventLog:
    def __init__(self, path: Path | str = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def append(
        self,
        task_id: str,
        event_type: EventType,
        payload: dict[str, Any],
        correlation_id: str | None = None,
    ) -> Event:
        """Write one event and commit it immediately (P4: record before side effects)."""
        created_at = datetime.now(UTC)
        text = json.dumps(payload, default=str)
        with self._conn:  # commits on success, rolls back on error
            cursor = self._conn.execute(
                "INSERT INTO events (task_id, type, payload, created_at, correlation_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (task_id, event_type.value, text, created_at.isoformat(), correlation_id),
            )
        if cursor.lastrowid is None:
            raise RuntimeError("event insert returned no row id")
        return Event(
            id=cursor.lastrowid,
            task_id=task_id,
            type=event_type,
            payload=json.loads(text),
            created_at=created_at,
            correlation_id=correlation_id,
        )

    def events(self, task_id: str, event_type: EventType | None = None) -> list[Event]:
        query = "SELECT * FROM events WHERE task_id = ?"
        params: list[Any] = [task_id]
        if event_type is not None:
            query += " AND type = ?"
            params.append(event_type.value)
        rows = self._conn.execute(query + " ORDER BY id", params).fetchall()
        return [_row_to_event(row) for row in rows]

    def unmatched_tool_calls(self, task_id: str) -> list[Event]:
        """TOOL_CALL events with no TOOL_RESULT. After a crash these are 'indeterminate'."""
        rows = self._conn.execute(
            "SELECT c.* FROM events c WHERE c.task_id = ? AND c.type = ? AND NOT EXISTS ("
            "  SELECT 1 FROM events r WHERE r.task_id = c.task_id AND r.type = ? "
            "  AND r.correlation_id = c.correlation_id) ORDER BY c.id",
            (task_id, EventType.TOOL_CALL.value, EventType.TOOL_RESULT.value),
        ).fetchall()
        return [_row_to_event(row) for row in rows]