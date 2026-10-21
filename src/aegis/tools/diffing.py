"""Small helpers shared by the tools that change files: a bounded unified diff and a hash."""

import difflib
import hashlib

MAX_DIFF_CHARS = 3000


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unified_diff(
    path: str, before: str, after: str, *, limit: int = MAX_DIFF_CHARS
) -> tuple[str, bool]:
    """A unified diff of two texts, cut to `limit` characters. Returns (diff, was_truncated).

    An empty diff means the two texts are identical.
    """
    lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    text = "".join(line if line.endswith("\n") else line + "\n" for line in lines)
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n...[diff truncated]\n", True
