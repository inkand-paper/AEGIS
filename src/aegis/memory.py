"""Append-only event log plus a checkpoint table, in SQLite (design doc sections 4.3, 5.5, 8).

The event log is the source of truth for "what actually happened". SQLite triggers make UPDATE
and DELETE on it fail, so even a bug in our own code cannot rewrite history. The checkpoint table
is the opposite: one mutable row per task saying "where we think we are". Keeping both lets
recovery compare reality (events) with belief (checkpoint).
"""

import json
import sqlite3
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from aegis.models import BudgetUsage, Event, EventType

# Schema versions. The number is stored in SQLite's `PRAGMA user_version`. A migration is never
# edited after it ships: a change to the schema is a NEW entry at the end of this list.
#
#   1  events (append-only, enforced by triggers) and checkpoints.
#   2  leases (one live process per task) and an index that speeds up P5 checks.
_MIGRATION_1 = """
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
CREATE TABLE IF NOT EXISTS checkpoints (
    task_id    TEXT PRIMARY KEY,
    snapshot   TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

_MIGRATION_2 = """
CREATE INDEX IF NOT EXISTS idx_events_corr ON events (task_id, type, correlation_id);
CREATE TABLE IF NOT EXISTS leases (
    task_id      TEXT PRIMARY KEY,
    owner        TEXT NOT NULL,
    acquired_at  REAL NOT NULL,
    heartbeat_at REAL NOT NULL
);
"""

MIGRATIONS: list[tuple[int, str]] = [(1, _MIGRATION_1), (2, _MIGRATION_2)]
LATEST_SCHEMA_VERSION = MIGRATIONS[-1][0]


class SchemaTooNew(Exception):
    """The database was written by a newer AEGIS. Opening it could damage it, so we refuse."""


class LeaseHeld(Exception):
    """Another live process is working on this task."""

    def __init__(self, task_id: str, owner: str, age_s: float) -> None:
        super().__init__(
            f"task {task_id} is being worked on by another process ({owner}, "
            f"last sign of life {age_s:.0f}s ago)"
        )
        self.owner = owner
        self.age_s = age_s


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        id=row["id"],
        task_id=row["task_id"],
        type=EventType(row["type"]),
        payload=json.loads(row["payload"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        correlation_id=row["correlation_id"],
    )


OPEN_ATTEMPTS = 6
OPEN_RETRY_DELAY_S = 0.25


class EventLog:
    def __init__(self, path: Path | str = ":memory:") -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._open()
        self._migrate()

    def _open(self) -> sqlite3.Connection:
        """Open the database. Right after a process was killed, Windows can briefly refuse the
        leftover WAL files ("disk I/O error"), and that is exactly when `resume` opens it, so a
        few short retries are part of crash recovery. Any other error is raised at once."""
        last: sqlite3.OperationalError | None = None
        for attempt in range(OPEN_ATTEMPTS):
            if attempt:
                time.sleep(OPEN_RETRY_DELAY_S)
            conn: sqlite3.Connection | None = None
            try:
                conn = sqlite3.connect(self._path)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA busy_timeout = 5000")
                if self._path != ":memory:":
                    # WAL lets `aegis status` read while a task is writing. FULL makes every
                    # committed event survive a power cut ("record before side effects").
                    conn.execute("PRAGMA journal_mode = WAL")
                    conn.execute("PRAGMA synchronous = FULL")
                return conn
            except sqlite3.OperationalError as exc:
                if conn is not None:
                    conn.close()
                if "disk i/o error" not in str(exc).lower():
                    raise
                last = exc
        assert last is not None
        raise last

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ schema versions

    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def _migrate(self) -> None:
        current = self.schema_version()
        if current > LATEST_SCHEMA_VERSION:
            raise SchemaTooNew(
                f"{self._path} has schema version {current}, but this AEGIS only knows up to "
                f"{LATEST_SCHEMA_VERSION}. Upgrade AEGIS instead of opening it with this one."
            )
        pending = [(version, sql) for version, sql in MIGRATIONS if version > current]
        if not pending:
            return
        has_data = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'events'"
        ).fetchone()
        if has_data and self._path != ":memory:":
            self._backup(f"{self._path}.bak-v{current}")
        for version, sql in pending:
            try:
                self._conn.executescript(
                    f"BEGIN;\n{sql}\nPRAGMA user_version = {version};\nCOMMIT;"
                )
            except sqlite3.Error:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

    def _backup(self, target: str) -> None:
        """A consistent copy of the database before a schema change (works with WAL too)."""
        copy = sqlite3.connect(target)
        try:
            self._conn.backup(copy)
        finally:
            copy.close()

    # ------------------------------------------------------------------ leases (one worker)

    def acquire_lease(
        self, task_id: str, owner: str, ttl_s: float, now: float | None = None
    ) -> None:
        """Take the right to work on a task, or raise LeaseHeld. A lease nobody has renewed for
        `ttl_s` seconds belongs to a dead process and can be taken over."""
        moment = time.time() if now is None else now
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO leases (task_id, owner, acquired_at, heartbeat_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(task_id) DO UPDATE SET "
                "owner = excluded.owner, acquired_at = excluded.acquired_at, "
                "heartbeat_at = excluded.heartbeat_at "
                "WHERE leases.owner = excluded.owner OR leases.heartbeat_at < ?",
                (task_id, owner, moment, moment, moment - ttl_s),
            )
        if cursor.rowcount == 0:
            row = self._conn.execute(
                "SELECT owner, heartbeat_at FROM leases WHERE task_id = ?", (task_id,)
            ).fetchone()
            raise LeaseHeld(task_id, row["owner"], max(0.0, moment - row["heartbeat_at"]))

    def renew_lease(self, task_id: str, owner: str, now: float | None = None) -> bool:
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE leases SET heartbeat_at = ? WHERE task_id = ? AND owner = ?",
                (time.time() if now is None else now, task_id, owner),
            )
        return cursor.rowcount == 1

    def release_lease(self, task_id: str, owner: str) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM leases WHERE task_id = ? AND owner = ?", (task_id, owner)
            )

    # ------------------------------------------------------------------ events (append-only)

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

    def last_effectful_result(self, task_id: str) -> dict[str, Any] | None:
        """The payload of the latest SUCCESSFUL tool result that changed something (or ran code).

        Used for duplicate suppression: a non-idempotent call is skipped only when it is the same
        call as the most recent change and nothing else has changed anything since.
        """
        row = self._conn.execute(
            "SELECT payload FROM events WHERE task_id = ? AND type = ? "
            "AND json_extract(payload, '$.success') = 1 "
            "AND json_extract(payload, '$.side_effects') IS NOT NULL "
            "AND json_extract(payload, '$.side_effects') != 'none' "
            "ORDER BY id DESC LIMIT 1",
            (task_id, EventType.TOOL_RESULT.value),
        ).fetchone()
        if row is None:
            return None
        payload: dict[str, Any] = json.loads(row["payload"])
        return payload

    def step_calls(self, task_id: str, step_id: str) -> list[tuple[Event, Event | None]]:
        """Every tool call made for one step, oldest first, each with its result (or None)."""
        results = {
            e.correlation_id: e
            for e in self.events(task_id, EventType.TOOL_RESULT)
            if e.correlation_id
        }
        return [
            (call, results.get(call.correlation_id or ""))
            for call in self.events(task_id, EventType.TOOL_CALL)
            if call.payload.get("step_id") == step_id
        ]

    def unmatched_tool_calls(self, task_id: str) -> list[Event]:
        """TOOL_CALL events with no TOOL_RESULT. After a crash these are 'indeterminate'."""
        rows = self._conn.execute(
            "SELECT c.* FROM events c WHERE c.task_id = ? AND c.type = ? AND NOT EXISTS ("
            "  SELECT 1 FROM events r WHERE r.task_id = c.task_id AND r.type = ? "
            "  AND r.correlation_id = c.correlation_id) ORDER BY c.id",
            (task_id, EventType.TOOL_CALL.value, EventType.TOOL_RESULT.value),
        ).fetchall()
        return [_row_to_event(row) for row in rows]

    def recompute_usage(self, task_id: str) -> BudgetUsage:
        """Rebuild usage from events alone. Compare with the checkpoint to detect drift.

        Wall-clock time cannot be rebuilt from events (idle time is not work), so it stays 0.
        """
        model_calls = 0
        tokens = 0
        dollars = Decimal("0")
        for event in self.events(task_id, EventType.MODEL_CALL):
            payload = event.payload
            model_calls += int(payload.get("attempts", 1))
            tokens += int(payload.get("input_tokens", 0)) + int(payload.get("output_tokens", 0))
            dollars += Decimal(str(payload.get("dollars", "0")))
        tool_calls = len(self.events(task_id, EventType.TOOL_CALL))
        return BudgetUsage(
            model_calls=model_calls, tool_calls=tool_calls, tokens_total=tokens, dollars=dollars
        )

    # ------------------------------------------------------------------ checkpoints (mutable)

    def save_checkpoint(self, task_id: str, snapshot: dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO checkpoints (task_id, snapshot, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(task_id) DO UPDATE SET "
                "snapshot = excluded.snapshot, updated_at = excluded.updated_at",
                (task_id, json.dumps(snapshot, default=str), datetime.now(UTC).isoformat()),
            )

    def load_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT snapshot FROM checkpoints WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        snapshot: dict[str, Any] = json.loads(row["snapshot"])
        return snapshot
