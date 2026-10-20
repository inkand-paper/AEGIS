"""Explicit tool registry. A tool exists only if it is registered here."""

from aegis.tools.base import ToolRegistry
from aegis.tools.calculator import CalculatorTool
from aegis.tools.filesystem import FsListTool, FsReadTool


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FsListTool())
    registry.register(FsReadTool())
    return registry