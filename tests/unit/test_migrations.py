import json
import sqlite3
from pathlib import Path

import pytest

import aegis.memory as memory
from aegis.memory import (
    LATEST_SCHEMA_VERSION,
    EventLog,
    LeaseHeld,
    SchemaTooNew,
)
from aegis.models import EventType


def tables(path: Path) -> set[str]:
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


def version(path: Path) -> int:
    with sqlite3.connect(path) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])


def make_v1_database(path: Path, *, stamp_version: bool = True) -> None:
    """A database exactly as Phase 2 created it: events and checkpoints, nothing else."""
    conn = sqlite3.connect(path)
    conn.executescript(memory._MIGRATION_1)
    conn.execute(
        "INSERT INTO events (task_id, type, payload, created_at, correlation_id) "
        "VALUES ('t1', 'task_created', '{\"objective\": \"old task\"}', '2026-10-01T00:00:00', NULL)"
    )
    conn.execute(
        "INSERT INTO checkpoints (task_id, snapshot, updated_at) "
        "VALUES ('t1', '{\"task\": {\"id\": \"t1\"}}', '2026-10-01T00:00:00')"
    )
    if stamp_version:
        conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- migrations


def test_a_new_database_is_created_at_the_latest_version(tmp_path: Path) -> None:
    path = tmp_path / "new.db"
    log = EventLog(path)
    assert log.schema_version() == LATEST_SCHEMA_VERSION == version(path)
    assert {"events", "checkpoints", "leases"} <= tables(path)
    assert not list(tmp_path.glob("*.bak-*"))  # nothing to back up


def test_upgrading_from_the_previous_schema_keeps_every_row(tmp_path: Path) -> None:
    """Phase 4 exit criterion: a migration test upgrades from the previous schema version."""
    path = tmp_path / "old.db"
    make_v1_database(path)
    assert "leases" not in tables(path)

    log = EventLog(path)

    assert log.schema_version() == LATEST_SCHEMA_VERSION
    assert "leases" in tables(path)
    old = log.events("t1")
    assert [e.type for e in old] == [EventType.TASK_CREATED]
    assert old[0].payload == {"objective": "old task"}
    assert log.load_checkpoint("t1") == {"task": {"id": "t1"}}
    new = log.append("t1", EventType.ERROR, {"note": "after upgrade"})
    assert new.id > old[0].id  # ids carry on, nothing was renumbered


def test_the_upgrade_leaves_a_backup_that_is_still_the_old_version(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    make_v1_database(path)
    EventLog(path)
    backup = tmp_path / "old.db.bak-v1"
    assert backup.exists()
    assert version(backup) == 1
    assert "leases" not in tables(backup)


def test_a_database_made_before_versioning_is_adopted(tmp_path: Path) -> None:
    path = tmp_path / "unstamped.db"
    make_v1_database(path, stamp_version=False)
    assert version(path) == 0
    log = EventLog(path)
    assert log.schema_version() == LATEST_SCHEMA_VERSION
    assert len(log.events("t1")) == 1
    assert (tmp_path / "unstamped.db.bak-v0").exists()


def test_opening_an_up_to_date_database_changes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite"
    EventLog(path).append("t", EventType.ERROR, {})
    EventLog(path)
    EventLog(path)
    assert version(path) == LATEST_SCHEMA_VERSION
    assert not list(tmp_path.glob("*.bak-*"))


def test_a_database_from_a_newer_aegis_is_refused_and_untouched(tmp_path: Path) -> None:
    path = tmp_path / "future.db"
    EventLog(path).close()
    with sqlite3.connect(path) as conn:
        conn.execute(f"PRAGMA user_version = {LATEST_SCHEMA_VERSION + 5}")
    with pytest.raises(SchemaTooNew, match="Upgrade AEGIS"):
        EventLog(path)
    assert version(path) == LATEST_SCHEMA_VERSION + 5


def test_a_failing_migration_rolls_back_completely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "db.sqlite"
    EventLog(path).close()
    broken = "CREATE TABLE half_done (x INTEGER);\nTHIS IS NOT SQL;"
    monkeypatch.setattr(memory, "MIGRATIONS", [*memory.MIGRATIONS, (3, broken)])
    monkeypatch.setattr(memory, "LATEST_SCHEMA_VERSION", 3)
    with pytest.raises(sqlite3.Error):
        EventLog(path)
    assert version(path) == 2
    assert "half_done" not in tables(path)


def test_the_event_log_is_still_append_only_after_an_upgrade(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    make_v1_database(path)
    EventLog(path)
    with sqlite3.connect(path) as conn, pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE events SET payload = '{}'")
    with sqlite3.connect(path) as conn, pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM events")


def test_migration_numbers_are_consecutive_from_one() -> None:
    assert [v for v, _ in memory.MIGRATIONS] == list(range(1, LATEST_SCHEMA_VERSION + 1))


def test_the_database_uses_wal_so_status_can_read_while_a_task_writes(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite"
    writer = EventLog(path)
    reader = EventLog(path)
    writer.append("t", EventType.ERROR, {"n": 1})
    assert len(reader.events("t")) == 1
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert json.loads(json.dumps(reader.events("t")[0].payload)) == {"n": 1}


# ---------------------------------------------------------------- leases


def test_a_lease_has_one_owner_at_a_time(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "db.sqlite")
    log.acquire_lease("t", "alice", ttl_s=30, now=1000)
    with pytest.raises(LeaseHeld) as info:
        log.acquire_lease("t", "bob", ttl_s=30, now=1010)
    assert info.value.owner == "alice"
    assert info.value.age_s == pytest.approx(10)
    log.acquire_lease("t", "alice", ttl_s=30, now=1011)  # the owner may take it again


def test_a_lease_nobody_renews_can_be_taken_over(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "db.sqlite")
    log.acquire_lease("t", "alice", ttl_s=30, now=1000)
    log.acquire_lease("t", "bob", ttl_s=30, now=1031)  # alice went quiet for over 30s
    assert not log.renew_lease("t", "alice", now=1032)  # and she has lost it
    assert log.renew_lease("t", "bob", now=1032)


def test_renewing_keeps_a_lease_alive(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "db.sqlite")
    log.acquire_lease("t", "alice", ttl_s=30, now=1000)
    assert log.renew_lease("t", "alice", now=1025)
    with pytest.raises(LeaseHeld):
        log.acquire_lease("t", "bob", ttl_s=30, now=1050)  # only 25s since the last renewal


def test_a_released_lease_is_free_and_only_its_owner_can_release_it(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "db.sqlite")
    log.acquire_lease("t", "alice", ttl_s=30, now=1000)
    log.release_lease("t", "bob")  # not the owner: nothing happens
    with pytest.raises(LeaseHeld):
        log.acquire_lease("t", "bob", ttl_s=30, now=1001)
    log.release_lease("t", "alice")
    log.acquire_lease("t", "bob", ttl_s=30, now=1002)


def test_leases_are_shared_between_connections_to_the_same_file(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite"
    EventLog(path).acquire_lease("t", "process-1", ttl_s=30, now=1000)
    with pytest.raises(LeaseHeld):
        EventLog(path).acquire_lease("t", "process-2", ttl_s=30, now=1001)


def test_leases_are_per_task(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "db.sqlite")
    log.acquire_lease("t1", "alice", ttl_s=30, now=1000)
    log.acquire_lease("t2", "bob", ttl_s=30, now=1000)
