"""Allowlisted program runners: shell_run_tests and shell_run_formatter (design doc section 6.3).

There is no general shell. Each tool builds ONE fixed command line (an argv list, never a shell
string) from a small, validated argument model, so the model cannot add flags or run something else.

Be clear about what this does NOT give you: running the tests runs code the model wrote, with your
user's permissions, and the process can still reach the network and your files. The workspace
check covers file tools only. That is why `shell_run_tests` has the side effect "execute", which
always needs approval unless the task was submitted with --allow-exec. Real isolation (a container)
arrives with Phase 6.
"""

import asyncio
import os
import re
import signal
import sys
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from aegis.tools.base import SideEffects, Tool, ToolContext, ToolError
from aegis.tools.diffing import sha256_hex

MAX_OUTPUT_CHARS = 6000
MAX_PATHS = 10
MAX_FORMAT_FILES = 500
RUN_TIMEOUT_S = 120.0

# A path argument can never start with "-", so it can never be read as a command line flag.
SafePath = Annotated[str, StringConstraints(min_length=1, max_length=200, pattern=r"^[^-]")]

_ENV_KEEP = (
    "PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT",
    "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL",
)  # fmt: skip


def clean_environment(workspace: Path) -> dict[str, str]:
    """The environment for a child process: no API keys, no tokens, nothing but the basics."""
    env = {name: os.environ[name] for name in _ENV_KEEP if os.environ.get(name)}
    env.update(
        HOME=str(workspace),
        USERPROFILE=str(workspace),
        PYTHONPATH=str(workspace),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTHONIOENCODING="utf-8",
        NO_COLOR="1",
    )
    return env


def _kill(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if sys.platform == "win32":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)  # the whole group, in case tests spawned more
    except (ProcessLookupError, PermissionError):
        pass


async def _read_tail(stream: asyncio.StreamReader, limit: int) -> tuple[bytes, int]:
    """Keep only the LAST `limit` bytes (a test summary is at the end). Returns (tail, total)."""
    kept = bytearray()
    total = 0
    while chunk := await stream.read(65536):
        total += len(chunk)
        kept += chunk
        if len(kept) > limit:
            del kept[: len(kept) - limit]
    return bytes(kept), total


async def run_program(argv: list[str], workspace: Path) -> tuple[int, str, bool]:
    """Run a fixed command in the workspace. Returns (exit code, output tail, was_truncated).

    If the call is cancelled or times out (the dispatcher does both by cancelling us), the
    process and its children are killed before the cancellation continues.
    """
    root = workspace.resolve()
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=root,
        env=clean_environment(root),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=sys.platform != "win32",
    )
    assert process.stdout is not None
    try:
        (tail, total), code = await asyncio.gather(
            _read_tail(process.stdout, MAX_OUTPUT_CHARS * 4), process.wait()
        )
    finally:
        _kill(process)
        await process.wait()
    text = tail.decode("utf-8", errors="replace")
    truncated = total > len(tail) or len(text) > MAX_OUTPUT_CHARS
    return code, text[-MAX_OUTPUT_CHARS:], truncated


# ---------------------------------------------------------------------------- shell_run_tests

_SUMMARY_LINE = re.compile(r"\bin \d+(?:\.\d+)?s\b")
_COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped)")


def parse_pytest_summary(output: str) -> dict[str, int]:
    """Read the counts from pytest's last summary line, for example '2 passed, 1 failed in 0.1s'."""
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for line in reversed(output.splitlines()):
        if _SUMMARY_LINE.search(line):
            for number, word in _COUNT.findall(line):
                counts["errors" if word.startswith("error") else word] = int(number)
            break
    return counts


class RunTestsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: list[SafePath] = Field(
        default_factory=list,
        max_length=MAX_PATHS,
        description="Test files or folders inside the workspace. Empty means everything.",
    )
    keyword: (
        Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_ ().-]{0,79}$")] | None
    ) = Field(default=None, description="Only run tests whose names match this (pytest -k).")


class RunTestsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exit_code: int
    passed: int
    failed: int
    errors: int
    skipped: int
    output: str
    output_truncated: bool


class RunTestsTool(Tool[RunTestsArgs, RunTestsResult]):
    name = "shell_run_tests"
    description = (
        "Run the project's pytest tests in the task workspace. Optionally pass test paths and a "
        "keyword. Reports exit code, counts and the end of the output."
    )
    arg_model = RunTestsArgs
    result_model = RunTestsResult
    side_effects = "execute"  # runs model-written code, see the module docstring
    requires_network = False
    idempotent = True
    path_fields = ("paths",)
    timeout_s = RUN_TIMEOUT_S

    async def execute(self, args: RunTestsArgs, ctx: ToolContext) -> RunTestsResult:
        argv = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
        if args.keyword:
            argv += ["-k", args.keyword]
        argv += ["--", *args.paths]
        code, output, truncated = await run_program(argv, ctx.workspace)
        counts = parse_pytest_summary(output)
        return RunTestsResult(exit_code=code, output=output, output_truncated=truncated, **counts)


# ---------------------------------------------------------------------------- shell_run_formatter


def _python_files(root: Path, targets: list[Path]) -> dict[str, str]:
    """Hash every .py file under the targets. Refuses symlinks, so ruff cannot be led outside."""
    hashes: dict[str, str] = {}
    for target in targets:
        candidates = [target] if target.is_file() else sorted(target.rglob("*"))
        for item in candidates:
            if item.is_symlink():
                raise ToolError(f"symlinks are not allowed in formatter targets: {item.name}")
            if item.is_file() and item.suffix == ".py":
                if len(hashes) >= MAX_FORMAT_FILES:
                    raise ToolError(f"more than {MAX_FORMAT_FILES} Python files; narrow the paths")
                hashes[item.relative_to(root).as_posix()] = sha256_hex(item.read_bytes())
    return hashes


class FormatArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: list[SafePath] = Field(
        default_factory=lambda: ["."],
        min_length=1,
        max_length=MAX_PATHS,
        description="Files or folders inside the workspace. '.' is the whole workspace.",
    )
    check_only: bool = Field(
        default=False, description="Only report what would change; do not rewrite files."
    )


class FormatResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exit_code: int
    changed: bool  # True when at least one file was rewritten
    files_changed: list[str]
    output: str
    output_truncated: bool


class FormatTool(Tool[FormatArgs, FormatResult]):
    name = "shell_run_formatter"
    description = (
        "Format Python files in the task workspace with ruff. Use check_only=true to see what "
        "would change without changing it."
    )
    arg_model = FormatArgs
    result_model = FormatResult
    side_effects = "workspace"
    requires_network = False
    idempotent = True
    path_fields = ("paths",)
    timeout_s = RUN_TIMEOUT_S

    def effective_side_effects(self, args: FormatArgs) -> SideEffects:
        return "none" if args.check_only else "workspace"

    async def execute(self, args: FormatArgs, ctx: ToolContext) -> FormatResult:
        root = ctx.workspace.resolve()
        targets = [ctx.resolve(path) for path in args.paths]
        before = _python_files(root, targets)
        argv = [sys.executable, "-m", "ruff", "format", "--isolated", "--no-cache"]
        if args.check_only:
            argv += ["--check", "--diff"]
        argv += ["--", *args.paths]
        code, output, truncated = await run_program(argv, ctx.workspace)
        after = _python_files(root, targets)
        changed = sorted(name for name, digest in after.items() if before.get(name) != digest)
        return FormatResult(
            exit_code=code,
            changed=bool(changed),
            files_changed=changed,
            output=output,
            output_truncated=truncated,
        )
