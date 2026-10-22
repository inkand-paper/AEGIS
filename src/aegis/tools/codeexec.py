"""code_exec: run a Python file from the workspace inside the Docker sandbox (Phase 6).

The model first writes the file with fs_write (which has its own approval), then asks to run it.
There is no inline-code argument on purpose: what runs is a file that is in the workspace, in the
event log and in the diff, never a string that skipped the write tool.

The tool is only offered to tasks submitted with --sandbox and it never runs on the host. It has
the side effect "execute", so it needs approval unless the task was submitted with --allow-exec.
The container has no network; see aegis.sandbox for the full list of limits and what they cannot
promise.
"""

import hashlib
import os
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from aegis.sandbox import Sandbox, SandboxUnavailable
from aegis.tools.base import Tool, ToolContext, ToolError, ToolUnavailable

MAX_SCRIPT_ARGS = 10
MAX_SNAPSHOT_FILES = 2000
MAX_HASHED_BYTES = 2 * 1024 * 1024
MAX_LISTED_CHANGES = 50

PyPath = Annotated[str, StringConstraints(min_length=4, max_length=200, pattern=r"^[^-].*\.py$")]
ScriptArg = Annotated[str, StringConstraints(max_length=200)]


class CodeExecArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: PyPath = Field(description="A .py file inside the workspace.")
    args: list[ScriptArg] = Field(
        default_factory=list, max_length=MAX_SCRIPT_ARGS, description="Command-line arguments."
    )


class CodeExecResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exit_code: int
    output: str
    output_truncated: bool
    timed_out: bool
    oom_killed: bool
    duration_s: float
    changed: bool  # True when the run created, changed or removed workspace files
    files_changed: list[str]
    note: str | None = None


def snapshot(root: Path) -> dict[str, str]:
    """A fingerprint of every file in the workspace, to see what a run changed."""
    found: dict[str, str] = {}
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        for name in sorted(files):
            if len(found) >= MAX_SNAPSHOT_FILES:
                return found
            full = Path(folder) / name
            key = full.relative_to(root).as_posix()
            try:
                if full.is_symlink():
                    found[key] = f"link:{os.readlink(full)}"
                    continue
                stat = full.stat()
                if stat.st_size <= MAX_HASHED_BYTES:
                    found[key] = hashlib.sha256(full.read_bytes()).hexdigest()
                else:
                    found[key] = f"big:{stat.st_size}:{stat.st_mtime_ns}"
            except OSError:
                found[key] = "unreadable"
    return found


class CodeExecTool(Tool[CodeExecArgs, CodeExecResult]):
    name = "code_exec"
    description = (
        "Run a Python file from the workspace in an isolated container with no network, a memory "
        "limit and a time limit. Write the file first with fs_write. Reports exit code and the "
        "end of the output, and which workspace files the run changed."
    )
    arg_model = CodeExecArgs
    result_model = CodeExecResult
    side_effects = "execute"
    requires_network = False
    idempotent = False  # a script may write files or append to them
    requires_sandbox = True
    path_fields = ("path",)
    timeout_s = 120.0

    def __init__(self, sandbox: Sandbox | None = None) -> None:
        self._sandbox = sandbox

    async def execute(self, args: CodeExecArgs, ctx: ToolContext) -> CodeExecResult:
        if self._sandbox is None:
            raise ToolUnavailable("this task wants the sandbox but none is configured")
        script = ctx.resolve(args.path)
        if not script.is_file():
            raise ToolError(f"no such file in the workspace: {args.path}")
        if script.is_symlink():
            raise ToolError("symlinks are not allowed as scripts")
        root = ctx.workspace.resolve()
        before = snapshot(root)
        try:
            ran = await self._sandbox.run(["python", args.path, *args.args], root)
        except SandboxUnavailable as exc:
            raise ToolUnavailable(str(exc)) from None
        after = snapshot(root)
        changed = sorted(n for n in before.keys() | after.keys() if before.get(n) != after.get(n))
        return CodeExecResult(
            exit_code=ran.exit_code,
            output=ran.output,
            output_truncated=ran.truncated,
            timed_out=ran.timed_out,
            oom_killed=ran.oom_killed,
            duration_s=ran.duration_s,
            changed=bool(changed),
            files_changed=changed[:MAX_LISTED_CHANGES],
            note=ran.note(),
        )
