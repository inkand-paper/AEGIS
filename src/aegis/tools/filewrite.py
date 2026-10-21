"""Tools that change files inside the workspace: fs_write and fs_edit (design doc section 6.3).

Both can run as a dry run (`dry_run=true`): the same checks and the same diff, but nothing is
written. A dry run has no side effect, so it needs no approval.
"""

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from aegis.tools.base import SideEffects, Tool, ToolContext, ToolError
from aegis.tools.diffing import sha256_hex, unified_diff

MAX_WRITE_CHARS = 200_000
MAX_EDIT_CHARS = 20_000
MAX_EDIT_FILE_BYTES = 1_000_000


class FileChangeResult(BaseModel):
    """What happened to one file. Shared by fs_write and fs_edit."""

    model_config = ConfigDict(extra="forbid")

    path: str
    created: bool
    changed: bool  # False when the new content equals the old content
    dry_run: bool
    size_bytes: int  # size the file has (or would have) afterwards
    sha256_before: str | None
    sha256_after: str
    diff: str
    diff_truncated: bool


def _read_existing(target: Path, shown: str) -> str | None:
    """The current text of a file, or None if it does not exist."""
    if target.is_dir():
        raise ToolError(f"{shown} is a directory")
    if not target.exists():
        return None
    if target.stat().st_size > MAX_EDIT_FILE_BYTES:
        raise ToolError(f"{shown} is larger than {MAX_EDIT_FILE_BYTES} bytes")
    data = target.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolError(f"{shown} is not a UTF-8 text file") from None


def _write_atomically(target: Path, data: bytes) -> None:
    """Write to a temp file in the same folder, then swap it in, so a crash leaves no half file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=target.parent, prefix=".aegis-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _finish(
    shown: str, target: Path, before: str | None, after: str, dry_run: bool
) -> FileChangeResult:
    data = after.encode("utf-8")
    changed = before != after
    if changed and not dry_run:
        _write_atomically(target, data)
    diff, truncated = unified_diff(shown, before or "", after)
    return FileChangeResult(
        path=shown,
        created=before is None,
        changed=changed,
        dry_run=dry_run,
        size_bytes=len(data),
        sha256_before=None if before is None else sha256_hex(before.encode("utf-8")),
        sha256_after=sha256_hex(data),
        diff=diff,
        diff_truncated=truncated,
    )


# ---------------------------------------------------------------------------- fs_write


class FsWriteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="File inside the workspace, for example 'src/app.py'.")
    content: str = Field(max_length=MAX_WRITE_CHARS, description="The complete new file content.")
    overwrite: bool = Field(
        default=False,
        description="Allow replacing a file that already exists. Prefer fs_edit for small changes.",
    )
    dry_run: bool = Field(default=False, description="Show the diff without writing anything.")


class FsWriteTool(Tool[FsWriteArgs, FileChangeResult]):
    name = "fs_write"
    description = (
        "Create a text file in the task workspace (parent folders are created). "
        "Replacing an existing file needs overwrite=true."
    )
    arg_model = FsWriteArgs
    result_model = FileChangeResult
    side_effects = "workspace"
    requires_network = False
    idempotent = False
    path_fields = ("path",)

    def effective_side_effects(self, args: FsWriteArgs) -> SideEffects:
        return "none" if args.dry_run else "workspace"

    async def execute(self, args: FsWriteArgs, ctx: ToolContext) -> FileChangeResult:
        target = ctx.resolve(args.path)
        if target == ctx.workspace.resolve():
            raise ToolError("cannot write to the workspace root itself")
        before = _read_existing(target, args.path)
        if before is not None and not args.overwrite:
            raise ToolError(f"{args.path} already exists. Use overwrite=true or fs_edit.")
        return _finish(args.path, target, before, args.content, args.dry_run)


# ---------------------------------------------------------------------------- fs_edit


class FsEditArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="An existing text file inside the workspace.")
    old: str = Field(
        min_length=1,
        max_length=MAX_EDIT_CHARS,
        description="The exact text to replace. It must occur exactly once in the file.",
    )
    new: str = Field(max_length=MAX_EDIT_CHARS, description="The replacement text.")
    dry_run: bool = Field(default=False, description="Show the diff without writing anything.")


class FsEditTool(Tool[FsEditArgs, FileChangeResult]):
    name = "fs_edit"
    description = (
        "Replace one exact piece of text in an existing workspace file. The old text must match "
        "exactly once, so include enough surrounding lines to make it unique."
    )
    arg_model = FsEditArgs
    result_model = FileChangeResult
    side_effects = "workspace"
    requires_network = False
    idempotent = False
    path_fields = ("path",)

    def effective_side_effects(self, args: FsEditArgs) -> SideEffects:
        return "none" if args.dry_run else "workspace"

    async def execute(self, args: FsEditArgs, ctx: ToolContext) -> FileChangeResult:
        target = ctx.resolve(args.path)
        before = _read_existing(target, args.path)
        if before is None:
            raise ToolError(f"path not found: {args.path} (use fs_write to create files)")
        count = before.count(args.old)
        if count == 0:
            raise ToolError(f"the old text was not found in {args.path}. Read the file again.")
        if count > 1:
            raise ToolError(
                f"the old text occurs {count} times in {args.path}. Include more surrounding lines."
            )
        after = before.replace(args.old, args.new, 1)
        return _finish(args.path, target, before, after, args.dry_run)
