import asyncio
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from aegis.executor import ExecutionResult, Executor
from aegis.memory import EventLog
from aegis.models import ToolCall, ToolOutcome
from aegis.tools import default_registry
from aegis.tools.base import ToolContext
from aegis.tools.shell import clean_environment, parse_pytest_summary

ALLOW = frozenset({"workspace", "execute"})

BUGGY = "def add(a, b):\n    return a - b\n"
TEST_ADD = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"


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
    executor: Executor | None = None,
) -> ExecutionResult:
    ctx = ToolContext(task_id="t", workspace=workspace, allowed_effects=allowed)
    tool_call = ToolCall(
        id=f"c{len(events.events('t'))}",
        task_id="t",
        tool_name=tool,
        arguments=arguments,
        requested_at=datetime.now(UTC),
    )
    runner = executor or Executor(default_registry(), events)
    return asyncio.run(runner.execute(tool_call, ctx))


# ---------------------------------------------------------------- the allowlist (exit criterion)


@pytest.mark.parametrize(
    "arguments",
    [
        {"command": "rm -rf /"},  # a field that does not exist
        {"args": ["--rootdir=/"]},
        {"paths": ["-x"]},  # would be read as a flag
        {"paths": ["--collect-only"]},
        {"paths": "tests"},  # wrong shape: must be a list
        {"keyword": "-x"},
        {"keyword": "a; rm -rf /"},
        {"keyword": "$(whoami)"},
        {"paths": [""]},
        {"paths": ["t.py"] * 11},
    ],
)
def test_run_tests_rejects_argument_shapes_outside_the_allowlist(
    events: EventLog, workspace: Path, arguments: dict[str, Any]
) -> None:
    result = call(events, workspace, "shell_run_tests", arguments)
    assert result.record.outcome is ToolOutcome.INVALID_ARGUMENTS


@pytest.mark.parametrize("path", ["../x", "/etc", "a/../../x"])
def test_run_tests_paths_must_stay_in_the_workspace(
    events: EventLog, workspace: Path, path: str
) -> None:
    result = call(events, workspace, "shell_run_tests", {"paths": [path]})
    assert result.record.outcome is ToolOutcome.POLICY_VIOLATION


def test_formatter_rejects_shapes_outside_the_allowlist(events: EventLog, workspace: Path) -> None:
    for arguments in ({"paths": ["--config=evil.toml"]}, {"flags": ["--help"]}, {"paths": []}):
        result = call(events, workspace, "shell_run_formatter", arguments)
        assert result.record.outcome is ToolOutcome.INVALID_ARGUMENTS


# ---------------------------------------------------------------- approval


def test_running_tests_needs_approval_even_when_writes_are_declared(
    events: EventLog, workspace: Path
) -> None:
    (workspace / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
    result = call(events, workspace, "shell_run_tests", {}, allowed=frozenset({"workspace"}))
    assert result.record.outcome is ToolOutcome.APPROVAL_REQUIRED


# ---------------------------------------------------------------- running tests


def test_a_small_project_fails_then_passes_after_an_edit_and_the_edit_has_a_diff(
    events: EventLog, workspace: Path
) -> None:
    """Phase 3 exit criterion: create a project, run its tests, get a non-empty diff."""
    assert call(events, workspace, "fs_write", {"path": "calc.py", "content": BUGGY}).record.success
    assert call(
        events, workspace, "fs_write", {"path": "test_calc.py", "content": TEST_ADD}
    ).record.success

    failing = call(events, workspace, "shell_run_tests", {})
    assert failing.record.outcome is ToolOutcome.OK  # the tool worked; the tests failed
    assert (failing.record.result or {})["exit_code"] == 1
    assert (failing.record.result or {})["failed"] == 1
    assert "assert -1 == 5" in (failing.record.result or {})["output"]

    edit = call(events, workspace, "fs_edit", {"path": "calc.py", "old": "a - b", "new": "a + b"})
    diff = (edit.record.result or {})["diff"]
    assert diff
    assert "-    return a - b" in diff and "+    return a + b" in diff

    passing = call(events, workspace, "shell_run_tests", {})
    data = passing.record.result or {}
    assert (data["exit_code"], data["passed"], data["failed"]) == (0, 1, 0)


def test_run_tests_can_target_a_file_and_a_keyword(events: EventLog, workspace: Path) -> None:
    (workspace / "test_two.py").write_text(
        "def test_one():\n    assert True\n\n\ndef test_two():\n    assert False\n",
        encoding="utf-8",
    )
    result = call(
        events, workspace, "shell_run_tests", {"paths": ["test_two.py"], "keyword": "one"}
    )
    data = result.record.result or {}
    assert (data["exit_code"], data["passed"], data["failed"]) == (0, 1, 0)


def test_no_tests_is_not_a_pass(events: EventLog, workspace: Path) -> None:
    result = call(events, workspace, "shell_run_tests", {})
    data = result.record.result or {}
    assert data["exit_code"] != 0
    assert data["passed"] == 0


def test_tests_run_with_a_clean_environment(
    events: EventLog, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AEGIS_API_KEY", "gsk_supersecretvalue123456")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    (workspace / "test_env.py").write_text(
        "import os\n\n\ndef test_no_secrets():\n"
        "    assert 'AEGIS_API_KEY' not in os.environ\n"
        "    assert 'GITHUB_TOKEN' not in os.environ\n",
        encoding="utf-8",
    )
    data = call(events, workspace, "shell_run_tests", {}).record.result or {}
    assert data["passed"] == 1, data["output"]


def test_clean_environment_keeps_only_the_basics(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AEGIS_API_KEY", "secret")
    env = clean_environment(workspace)
    assert "AEGIS_API_KEY" not in env
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX only")
def test_a_hanging_test_is_killed_at_the_timeout(events: EventLog, workspace: Path) -> None:
    (workspace / "test_hang.py").write_text(
        "import time\n\n\ndef test_hang():\n    time.sleep(60)\n", encoding="utf-8"
    )
    registry = default_registry()
    registry.get("shell_run_tests").timeout_s = 1.0
    started = time.monotonic()
    result = call(events, workspace, "shell_run_tests", {}, executor=Executor(registry, events))
    elapsed = time.monotonic() - started
    assert result.record.outcome is ToolOutcome.TIMEOUT
    assert elapsed < 10


def test_huge_output_is_cut_to_the_tail(events: EventLog, workspace: Path) -> None:
    (workspace / "test_noisy.py").write_text(
        "def test_noisy():\n    print('x' * 500_000)\n    assert False, 'END-MARKER'\n",
        encoding="utf-8",
    )
    data = call(events, workspace, "shell_run_tests", {}).record.result or {}
    assert data["output_truncated"] is True
    assert len(data["output"]) <= 6000
    assert "failed" in data["output"]  # the summary is at the end, so the tail keeps it


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("....\n4 passed in 0.02s", {"passed": 4, "failed": 0, "errors": 0, "skipped": 0}),
        (
            "1 failed, 2 passed, 1 skipped in 1.50s",
            {"passed": 2, "failed": 1, "errors": 0, "skipped": 1},
        ),
        ("2 errors in 0.10s", {"passed": 0, "failed": 0, "errors": 2, "skipped": 0}),
        ("no tests ran in 0.01s", {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}),
        ("garbage", {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}),
    ],
)
def test_pytest_summary_parsing(text: str, expected: dict[str, int]) -> None:
    assert parse_pytest_summary(text) == expected


# ---------------------------------------------------------------- formatter

UGLY = "def f( a,b ):\n  return a+b\n"


def test_formatter_rewrites_files_and_reports_which(events: EventLog, workspace: Path) -> None:
    (workspace / "ugly.py").write_text(UGLY, encoding="utf-8")
    data = call(events, workspace, "shell_run_formatter", {}).record.result or {}
    assert data["exit_code"] == 0
    assert data["files_changed"] == ["ugly.py"]
    assert (workspace / "ugly.py").read_text(encoding="utf-8") == "def f(a, b):\n    return a + b\n"


def test_formatter_check_only_changes_nothing_and_needs_no_approval(
    events: EventLog, workspace: Path
) -> None:
    (workspace / "ugly.py").write_text(UGLY, encoding="utf-8")
    result = call(
        events, workspace, "shell_run_formatter", {"check_only": True}, allowed=frozenset()
    )
    data = result.record.result or {}
    assert result.record.outcome is ToolOutcome.OK
    assert data["exit_code"] == 1  # ruff: "would reformat"
    assert data["changed"] is False
    assert (workspace / "ugly.py").read_text(encoding="utf-8") == UGLY


def test_formatter_rewriting_needs_approval(events: EventLog, workspace: Path) -> None:
    (workspace / "ugly.py").write_text(UGLY, encoding="utf-8")
    result = call(events, workspace, "shell_run_formatter", {}, allowed=frozenset())
    assert result.record.outcome is ToolOutcome.APPROVAL_REQUIRED
    assert (workspace / "ugly.py").read_text(encoding="utf-8") == UGLY


@pytest.mark.skipif(sys.platform == "win32" and not os.environ.get("CI"), reason="needs symlinks")
def test_formatter_refuses_symlinks_so_it_cannot_reach_outside_files(
    events: EventLog, workspace: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text(UGLY, encoding="utf-8")
    try:
        (workspace / "link.py").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow creating symlinks")
    result = call(events, workspace, "shell_run_formatter", {})
    assert result.record.outcome in {ToolOutcome.TOOL_ERROR, ToolOutcome.POLICY_VIOLATION}
    assert outside.read_text(encoding="utf-8") == UGLY
