"""The tool contract (design doc section 6) and the workspace sandbox (principle P6)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

# What a tool call can do to the world, from harmless to most dangerous:
#   none       reads only
#   workspace  creates or changes files inside the task workspace
#   execute    runs a program on this computer (for example the project's tests)
#   external   touches anything outside the workspace
SideEffects = Literal["none", "workspace", "execute", "external"]


class PolicyError(Exception):
    """A call was refused by policy (for example a path outside the workspace)."""


class ToolError(Exception):
    """A tool ran but could not do what was asked (file missing, bad value, ...).

    The message is sent back to the model, so write it to help the model correct itself.
    """


class ToolNotFound(KeyError):
    """No tool is registered under that name."""


@dataclass(frozen=True)
class ToolContext:
    """What a tool is allowed to know: which task it runs for and where its sandbox is."""

    task_id: str
    workspace: Path
    # Side-effect classes the task declared up front (for example {"workspace"}). A call whose
    # effect is not in this set needs a human approval before it runs.
    allowed_effects: frozenset[str] = frozenset()

    def resolve(self, relative: str) -> Path:
        """Turn a workspace-relative path into a real path, or raise PolicyError.

        Rejects absolute paths, `..` escapes and symlinks that lead outside the workspace.
        `Path.resolve()` follows symlinks, so a link pointing outside ends up outside the root.
        """
        if "\x00" in relative:
            raise PolicyError("path contains a NUL byte")
        candidate = Path(relative)
        if candidate.is_absolute() or candidate.drive or candidate.root:
            raise PolicyError(f"absolute paths are not allowed: {relative!r}")
        root = self.workspace.resolve()
        resolved = (root / candidate).resolve()
        if not resolved.is_relative_to(root):
            raise PolicyError(f"path escapes the workspace: {relative!r}")
        return resolved


class Tool[A: BaseModel, R: BaseModel](ABC):
    """Base class for every tool. `A` is the argument model, `R` is the result model."""

    name: str
    description: str
    arg_model: type[A]
    result_model: type[R]
    side_effects: SideEffects
    requires_network: bool
    idempotent: bool
    # Names of argument fields that hold workspace-relative paths. The dispatcher checks
    # every one of them against the workspace BEFORE the tool runs.
    # (a field may hold one path or a list of paths)
    path_fields: tuple[str, ...] = ()
    # Overrides the dispatcher's default timeout. Set by slow tools such as the test runner.
    timeout_s: float | None = None

    def effective_side_effects(self, args: A) -> SideEffects:
        """The side effect of THIS call. Tools override it so a dry run counts as harmless."""
        return self.side_effects

    @abstractmethod
    async def execute(self, args: A, ctx: ToolContext) -> R: ...


class ToolRegistry:
    """Explicit registration only. There is no auto-discovery (it hides the attack surface)."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any, Any]] = {}

    def register(self, tool: Tool[Any, Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any, Any]:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFound(name) from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def list_for_model(self) -> list[dict[str, Any]]:
        """The tool list in the neutral format that ModelClient.generate() accepts."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.arg_model.model_json_schema(),
            }
            for tool in self._tools.values()
        ]
