"""Explicit tool registry. A tool exists only if it is registered here."""

from aegis.tools.base import ToolRegistry
from aegis.tools.browser import browser_tools
from aegis.tools.calculator import CalculatorTool
from aegis.tools.filesystem import FsListTool, FsReadTool
from aegis.tools.filewrite import FsEditTool, FsWriteTool
from aegis.tools.shell import FormatTool, RunTestsTool


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FsListTool())
    registry.register(FsReadTool())
    registry.register(FsWriteTool())
    registry.register(FsEditTool())
    registry.register(RunTestsTool())
    registry.register(FormatTool())
    for tool in browser_tools():
        registry.register(tool)
    return registry
