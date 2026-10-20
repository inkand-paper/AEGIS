import asyncio
from pathlib import Path

import pytest

from aegis.tools.base import PolicyError, ToolContext, ToolError
from aegis.tools.filesystem import FsListArgs, FsListTool, FsReadArgs, FsReadTool


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    workspace = tmp_path / "workspace"
    (workspace / "notes").mkdir(parents=True)
    (workspace / "hello.txt").write_text("hello world", encoding="utf-8")
    (workspace / "notes" / "todo.txt").write_text("buy milk", encoding="utf-8")
    (workspace / "blob.bin").write_bytes(b"\x00\x01\x02")
    (tmp_path / "secret.txt").write_text("outside the workspace", encoding="utf-8")
    return ToolContext(task_id="t", workspace=workspace)


def fs_list(ctx: ToolContext, path: str = ".") -> list[tuple[str, str]]:
    result = asyncio.run(FsListTool().execute(FsListArgs(path=path), ctx))
    return [(e.name, e.kind) for e in result.entries]


def fs_read(ctx: ToolContext, path: str, max_bytes: int = 20_000):  # type: ignore[no-untyped-def]
    return asyncio.run(FsReadTool().execute(FsReadArgs(path=path, max_bytes=max_bytes), ctx))


def test_list_root_and_subdirectory(ctx: ToolContext) -> None:
    assert fs_list(ctx) == [("blob.bin", "file"), ("hello.txt", "file"), ("notes", "dir")]
    assert fs_list(ctx, "notes") == [("todo.txt", "file")]


def test_read_file(ctx: ToolContext) -> None:
    result = fs_read(ctx, "notes/todo.txt")
    assert (result.content, result.size_bytes, result.truncated) == ("buy milk", 8, False)


def test_read_truncates_at_max_bytes(ctx: ToolContext) -> None:
    result = fs_read(ctx, "hello.txt", max_bytes=5)
    assert (result.content, result.size_bytes, result.truncated) == ("hello", 11, True)


@pytest.mark.parametrize("path", ["missing.txt", "notes/missing.txt"])
def test_read_missing_file_is_a_tool_error(ctx: ToolContext, path: str) -> None:
    with pytest.raises(ToolError, match="not found"):
        fs_read(ctx, path)


def test_read_directory_and_list_file_are_tool_errors(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="directory"):
        fs_read(ctx, "notes")
    with pytest.raises(ToolError, match="not a directory"):
        fs_list(ctx, "hello.txt")


def test_binary_files_are_refused(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="binary"):
        fs_read(ctx, "blob.bin")


@pytest.mark.parametrize(
    "path",
    ["../secret.txt", "../../etc/passwd", "notes/../../secret.txt", "/etc/passwd", "bad\x00name"],
)
def test_paths_outside_the_workspace_are_rejected(ctx: ToolContext, path: str) -> None:
    with pytest.raises(PolicyError):
        ctx.resolve(path)
    with pytest.raises(PolicyError):
        fs_read(ctx, path)
    with pytest.raises(PolicyError):
        fs_list(ctx, path)


def test_dotdot_that_stays_inside_is_allowed(ctx: ToolContext) -> None:
    assert fs_read(ctx, "notes/../hello.txt").content == "hello world"


def test_symlink_escape_is_rejected(ctx: ToolContext, tmp_path: Path) -> None:
    link = ctx.workspace / "escape.txt"
    try:
        link.symlink_to(tmp_path / "secret.txt")
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow creating symlinks")
    with pytest.raises(PolicyError):
        fs_read(ctx, "escape.txt")
    # Listing shows the link as a link and never follows it.
    assert ("escape.txt", "symlink") in fs_list(ctx)


def test_directory_symlink_escape_is_rejected(ctx: ToolContext, tmp_path: Path) -> None:
    link = ctx.workspace / "outside_dir"
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow creating symlinks")
    with pytest.raises(PolicyError):
        fs_list(ctx, "outside_dir")
    with pytest.raises(PolicyError):
        fs_read(ctx, "outside_dir/secret.txt")


def test_listing_is_capped(ctx: ToolContext) -> None:
    for number in range(250):
        (ctx.workspace / f"f{number:03}.txt").write_text("x", encoding="utf-8")
    result = asyncio.run(FsListTool().execute(FsListArgs(), ctx))
    assert len(result.entries) == 200
    assert result.truncated is True