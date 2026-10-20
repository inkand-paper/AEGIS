"""Read-only filesystem tools. Every path is resolved against the task workspace (P6)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from aegis.tools.base import Tool, ToolContext, ToolError

MAX_LIST_ENTRIES = 200
DEFAULT_READ_BYTES = 20_000
MAX_READ_BYTES = 100_000


class FsListArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(default=".", description="Directory inside the workspace. '.' is the root.")


class FsEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["file", "dir", "symlink", "other"]
    size_bytes: int | None = None


class FsListResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    entries: list[FsEntry]
    truncated: bool


class FsListTool(Tool[FsListArgs, FsListResult]):
    name = "fs_list"
    description = "List the files and folders in a directory of the task workspace."
    arg_model = FsListArgs
    result_model = FsListResult
    side_effects = "none"
    requires_network = False
    idempotent = True
    path_fields = ("path",)

    async def execute(self, args: FsListArgs, ctx: ToolContext) -> FsListResult:
        target = ctx.resolve(args.path)
        if not target.exists():
            raise ToolError(f"path not found: {args.path}")
        if not target.is_dir():
            raise ToolError(f"not a directory: {args.path} (use fs_read for files)")

        entries: list[FsEntry] = []
        children = sorted(target.iterdir(), key=lambda p: p.name)
        for child in children[:MAX_LIST_ENTRIES]:
            if child.is_symlink():  # never follow links: they could point outside
                entries.append(FsEntry(name=child.name, kind="symlink"))
            elif child.is_dir():
                entries.append(FsEntry(name=child.name, kind="dir"))
            elif child.is_file():
                entries.append(
                    FsEntry(name=child.name, kind="file", size_bytes=child.stat().st_size)
                )
            else:
                entries.append(FsEntry(name=child.name, kind="other"))
        return FsListResult(
            path=args.path, entries=entries, truncated=len(children) > MAX_LIST_ENTRIES
        )


class FsReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="File inside the workspace, for example 'notes/todo.txt'.")
    max_bytes: int = Field(
        default=DEFAULT_READ_BYTES,
        gt=0,
        le=MAX_READ_BYTES,
        description="Read at most this many bytes.",
    )


class FsReadResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    content: str
    size_bytes: int
    truncated: bool


class FsReadTool(Tool[FsReadArgs, FsReadResult]):
    name = "fs_read"
    description = "Read a text file from the task workspace."
    arg_model = FsReadArgs
    result_model = FsReadResult
    side_effects = "none"
    requires_network = False
    idempotent = True
    path_fields = ("path",)

    async def execute(self, args: FsReadArgs, ctx: ToolContext) -> FsReadResult:
        target = ctx.resolve(args.path)
        if not target.exists():
            raise ToolError(f"path not found: {args.path}")
        if target.is_dir():
            raise ToolError(f"{args.path} is a directory (use fs_list)")
        with target.open("rb") as handle:
            data = handle.read(args.max_bytes + 1)
        if b"\x00" in data:
            raise ToolError(f"{args.path} looks like a binary file, not text")
        truncated = len(data) > args.max_bytes
        return FsReadResult(
            path=args.path,
            content=data[: args.max_bytes].decode("utf-8", errors="replace"),
            size_bytes=target.stat().st_size,
            truncated=truncated,
        )