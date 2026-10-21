"""Duplicate suppression for non-idempotent calls (design doc 5.5, Phase 4)."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from aegis.executor import ExecutionResult, Executor
from aegis.memory import EventLog
from aegis.models import EventType, ToolCall, ToolOutcome
from aegis.tools import default_registry
from aegis.tools.base import ToolContext

ALLOW = frozenset({"workspace", "execute"})


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
def events() -> EventLog:
    return EventLog()


def call(
    events: EventLog,
    workspace: Path,
    tool: str,
    arguments: dict[str, Any],
    *,
    step_id: str | None = "s1",
) -> ExecutionResult:
    ctx = ToolContext(task_id="t", workspace=workspace, allowed_effects=ALLOW)
    tool_call = ToolCall(
        id=f"c{len(events.events('t'))}",
        task_id="t",
        step_id=step_id,
        tool_name=tool,
        arguments=arguments,
        requested_at=datetime.now(UTC),
    )
    return asyncio.run(Executor(default_registry(), events).execute(tool_call, ctx))


def results(events: EventLog) -> list[dict[str, Any]]:
    return [e.payload for e in events.events("t", EventType.TOOL_RESULT)]


EDIT = {"path": "f.txt", "old": "one", "new": "two"}


def test_a_successful_change_records_its_key_and_effect(events: EventLog, workspace: Path) -> None:
    (workspace / "f.txt").write_text("one", encoding="utf-8")
    result = call(events, workspace, "fs_edit", EDIT)
    payload = results(events)[0]
    assert payload["side_effects"] == "workspace"
    assert payload["idempotency_key"] and len(payload["idempotency_key"]) == 64
    assert payload["deduplicated"] is False
    assert result.record.call.idempotency_key == payload["idempotency_key"]


def test_repeating_the_same_change_is_not_applied_twice(events: EventLog, workspace: Path) -> None:
    (workspace / "f.txt").write_text("one one", encoding="utf-8")
    ctx_args = {"path": "f.txt", "old": "one one", "new": "one"}
    first = call(events, workspace, "fs_edit", ctx_args)
    second = call(events, workspace, "fs_edit", ctx_args)  # the model asks for it again

    assert first.record.success and second.record.success
    assert second.record.result == first.record.result  # the recorded result comes back
    assert (workspace / "f.txt").read_text(encoding="utf-8") == "one"  # applied exactly once
    assert [r["deduplicated"] for r in results(events)] == [False, True]
    assert results(events)[0]["idempotency_key"] == results(events)[1]["idempotency_key"]


def test_a_different_call_is_never_suppressed(events: EventLog, workspace: Path) -> None:
    (workspace / "f.txt").write_text("one", encoding="utf-8")
    call(events, workspace, "fs_edit", EDIT)
    other = call(events, workspace, "fs_edit", {"path": "f.txt", "old": "two", "new": "three"})
    assert other.record.success
    assert (workspace / "f.txt").read_text(encoding="utf-8") == "three"


def test_a_repeat_after_another_change_runs_again(events: EventLog, workspace: Path) -> None:
    """A to B, B to A, A to B again is three real edits, not a duplicate."""
    (workspace / "f.txt").write_text("A", encoding="utf-8")
    a_to_b = {"path": "f.txt", "old": "A", "new": "B"}
    b_to_a = {"path": "f.txt", "old": "B", "new": "A"}
    call(events, workspace, "fs_edit", a_to_b)
    call(events, workspace, "fs_edit", b_to_a)
    third = call(events, workspace, "fs_edit", a_to_b)
    assert (workspace / "f.txt").read_text(encoding="utf-8") == "B"
    assert third.record.success
    assert [r["deduplicated"] for r in results(events)] == [False, False, False]


def test_running_code_in_between_resets_suppression(events: EventLog, workspace: Path) -> None:
    """Tests (or a formatter) can change files, so an earlier identical write no longer counts."""
    write = {"path": "a.txt", "content": "x\n", "overwrite": True}
    call(events, workspace, "fs_write", write)
    (workspace / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
    call(events, workspace, "shell_run_tests", {})
    (workspace / "a.txt").write_text("changed by something else", encoding="utf-8")
    again = call(events, workspace, "fs_write", write)
    assert again.record.success
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "x\n"  # it really ran
    assert results(events)[-1]["deduplicated"] is False


def test_the_same_call_in_another_step_is_not_a_duplicate(
    events: EventLog, workspace: Path
) -> None:
    write = {"path": "a.txt", "content": "x\n", "overwrite": True}
    call(events, workspace, "fs_write", write, step_id="s1")
    (workspace / "a.txt").write_text("tampered", encoding="utf-8")
    call(events, workspace, "fs_write", write, step_id="s2")
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "x\n"
    assert [r["deduplicated"] for r in results(events)] == [False, False]


def test_failed_changes_leave_no_key_and_can_be_retried(events: EventLog, workspace: Path) -> None:
    (workspace / "f.txt").write_text("nothing to match", encoding="utf-8")
    failed = call(events, workspace, "fs_edit", EDIT)
    assert failed.record.outcome is ToolOutcome.TOOL_ERROR
    assert results(events)[0]["idempotency_key"] is None
    (workspace / "f.txt").write_text("one", encoding="utf-8")
    assert call(events, workspace, "fs_edit", EDIT).record.success


def test_dry_runs_and_reads_have_no_key(events: EventLog, workspace: Path) -> None:
    (workspace / "f.txt").write_text("one", encoding="utf-8")
    call(events, workspace, "fs_edit", {**EDIT, "dry_run": True})
    call(events, workspace, "fs_read", {"path": "f.txt"})
    assert [r["idempotency_key"] for r in results(events)] == [None, None]
    assert [r["side_effects"] for r in results(events)] == ["none", "none"]


def test_every_suppressed_call_still_has_a_call_and_a_result_event(
    events: EventLog, workspace: Path
) -> None:
    (workspace / "f.txt").write_text("one one", encoding="utf-8")
    args = {"path": "f.txt", "old": "one one", "new": "one"}
    call(events, workspace, "fs_edit", args)
    call(events, workspace, "fs_edit", args)
    assert len(events.events("t", EventType.TOOL_CALL)) == 2
    assert events.unmatched_tool_calls("t") == []
