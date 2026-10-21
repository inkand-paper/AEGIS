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

ALLOW = frozenset({"workspace"})


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
    allowed: frozenset[str] = ALLOW,
) -> ExecutionResult:
    ctx = ToolContext(task_id="t", workspace=workspace, allowed_effects=allowed)
    tool_call = ToolCall(
        id=f"c{len(events.events('t'))}",
        task_id="t",
        tool_name=tool,
        arguments=arguments,
        requested_at=datetime.now(UTC),
    )
    return asyncio.run(Executor(default_registry(), events).execute(tool_call, ctx))


# ---------------------------------------------------------------- fs_write


def test_write_creates_a_file_and_parent_folders(events: EventLog, workspace: Path) -> None:
    result = call(events, workspace, "fs_write", {"path": "src/app.py", "content": "x = 1\n"})
    assert result.record.outcome is ToolOutcome.OK
    assert (workspace / "src" / "app.py").read_text(encoding="utf-8") == "x = 1\n"
    data = result.record.result or {}
    assert data["created"] is True and data["changed"] is True
    assert "+x = 1" in data["diff"]
    assert data["sha256_before"] is None


def test_write_refuses_to_replace_a_file_unless_told(events: EventLog, workspace: Path) -> None:
    (workspace / "a.txt").write_text("old", encoding="utf-8")
    result = call(events, workspace, "fs_write", {"path": "a.txt", "content": "new"})
    assert result.record.outcome is ToolOutcome.TOOL_ERROR
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "old"

    again = call(
        events, workspace, "fs_write", {"path": "a.txt", "content": "new", "overwrite": True}
    )
    assert again.record.outcome is ToolOutcome.OK
    assert (workspace / "a.txt").read_text(encoding="utf-8") == "new"


def test_writing_identical_content_reports_no_change(events: EventLog, workspace: Path) -> None:
    (workspace / "a.txt").write_text("same", encoding="utf-8")
    result = call(
        events, workspace, "fs_write", {"path": "a.txt", "content": "same", "overwrite": True}
    )
    assert (result.record.result or {})["changed"] is False
    assert (result.record.result or {})["diff"] == ""


def test_write_needs_approval_unless_the_task_declared_it(
    events: EventLog, workspace: Path
) -> None:
    result = call(
        events, workspace, "fs_write", {"path": "a.txt", "content": "x"}, allowed=frozenset()
    )
    assert result.record.outcome is ToolOutcome.APPROVAL_REQUIRED
    assert not (workspace / "a.txt").exists()


def test_a_dry_run_writes_nothing_and_needs_no_approval(events: EventLog, workspace: Path) -> None:
    result = call(
        events,
        workspace,
        "fs_write",
        {"path": "a.txt", "content": "x\n", "dry_run": True},
        allowed=frozenset(),
    )
    assert result.record.outcome is ToolOutcome.OK
    assert (result.record.result or {})["dry_run"] is True
    assert "+x" in (result.record.result or {})["diff"]
    assert not (workspace / "a.txt").exists()


@pytest.mark.parametrize("path", ["../out.txt", "/etc/evil", "a/../../out.txt", ".", ""])
def test_write_cannot_leave_the_workspace_or_hit_the_root(
    events: EventLog, workspace: Path, path: str
) -> None:
    result = call(events, workspace, "fs_write", {"path": path, "content": "x"})
    assert result.record.outcome in {ToolOutcome.POLICY_VIOLATION, ToolOutcome.TOOL_ERROR}
    assert not (workspace.parent / "out.txt").exists()


def test_write_through_a_symlink_to_the_outside_is_rejected(
    events: EventLog, workspace: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (workspace / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow creating symlinks")
    result = call(events, workspace, "fs_write", {"path": "link/pwned.txt", "content": "x"})
    assert result.record.outcome is ToolOutcome.POLICY_VIOLATION
    assert not (outside / "pwned.txt").exists()
    recorded = events.events("t", EventType.TOOL_RESULT)[-1]
    assert recorded.payload["outcome"] == "policy_violation"


def test_oversized_content_is_rejected(events: EventLog, workspace: Path) -> None:
    result = call(events, workspace, "fs_write", {"path": "big.txt", "content": "x" * 200_001})
    assert result.record.outcome is ToolOutcome.INVALID_ARGUMENTS


def test_no_temp_files_are_left_behind(events: EventLog, workspace: Path) -> None:
    call(events, workspace, "fs_write", {"path": "a.txt", "content": "x"})
    assert [p.name for p in workspace.iterdir()] == ["a.txt"]


# ---------------------------------------------------------------- fs_edit


def test_edit_replaces_one_exact_match_and_returns_a_diff(
    events: EventLog, workspace: Path
) -> None:
    (workspace / "m.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    result = call(events, workspace, "fs_edit", {"path": "m.py", "old": "a - b", "new": "a + b"})
    assert result.record.outcome is ToolOutcome.OK
    assert (workspace / "m.py").read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"
    data = result.record.result or {}
    assert "-    return a - b" in data["diff"] and "+    return a + b" in data["diff"]
    assert data["sha256_before"] != data["sha256_after"]


@pytest.mark.parametrize(
    ("content", "old", "message"),
    [("abc", "zzz", "not found"), ("aa aa", "aa", "2 times")],
)
def test_edit_needs_exactly_one_match(
    events: EventLog, workspace: Path, content: str, old: str, message: str
) -> None:
    (workspace / "f.txt").write_text(content, encoding="utf-8")
    result = call(events, workspace, "fs_edit", {"path": "f.txt", "old": old, "new": "x"})
    assert result.record.outcome is ToolOutcome.TOOL_ERROR
    assert message in (result.record.error or "")
    assert (workspace / "f.txt").read_text(encoding="utf-8") == content


def test_edit_of_a_missing_file_is_an_error_not_a_create(events: EventLog, workspace: Path) -> None:
    result = call(events, workspace, "fs_edit", {"path": "nope.txt", "old": "a", "new": "b"})
    assert result.record.outcome is ToolOutcome.TOOL_ERROR
    assert not (workspace / "nope.txt").exists()


def test_edit_refuses_binary_files(events: EventLog, workspace: Path) -> None:
    (workspace / "b.bin").write_bytes(b"\xff\xfe\x00\x01")
    result = call(events, workspace, "fs_edit", {"path": "b.bin", "old": "a", "new": "b"})
    assert result.record.outcome is ToolOutcome.TOOL_ERROR


def test_edit_dry_run_leaves_the_file_alone(events: EventLog, workspace: Path) -> None:
    (workspace / "f.txt").write_text("one", encoding="utf-8")
    result = call(
        events,
        workspace,
        "fs_edit",
        {"path": "f.txt", "old": "one", "new": "two", "dry_run": True},
        allowed=frozenset(),
    )
    assert result.record.outcome is ToolOutcome.OK
    assert (workspace / "f.txt").read_text(encoding="utf-8") == "one"


def test_edit_keeps_windows_line_endings(events: EventLog, workspace: Path) -> None:
    (workspace / "w.txt").write_bytes(b"a\r\nb\r\nc\r\n")
    call(events, workspace, "fs_edit", {"path": "w.txt", "old": "b", "new": "B"})
    assert (workspace / "w.txt").read_bytes() == b"a\r\nB\r\nc\r\n"


def test_a_large_diff_is_truncated_for_the_model(events: EventLog, workspace: Path) -> None:
    before = "".join(f"line {i}\n" for i in range(2000))
    (workspace / "big.txt").write_text(before, encoding="utf-8")
    result = call(
        events,
        workspace,
        "fs_write",
        {"path": "big.txt", "content": "changed\n", "overwrite": True},
    )
    data = result.record.result or {}
    assert data["diff_truncated"] is True
    assert len(data["diff"]) < 3200
