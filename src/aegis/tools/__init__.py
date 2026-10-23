"""Explicit tool registry. A tool exists only if it is registered here."""

from aegis.desktop import DesktopBackend
from aegis.images import ImageProvider, ProceduralProvider
from aegis.sandbox import Sandbox
from aegis.tools.base import ToolRegistry
from aegis.tools.browser import browser_tools
from aegis.tools.calculator import CalculatorTool
from aegis.tools.codeexec import CodeExecTool
from aegis.tools.desktop import desktop_tools
from aegis.tools.filesystem import FsListTool, FsReadTool
from aegis.tools.filewrite import FsEditTool, FsWriteTool
from aegis.tools.image import ImageGenerateTool
from aegis.tools.shell import FormatTool, RunTestsTool


def default_registry(
    sandbox: Sandbox | None = None,
    desktop: DesktopBackend | None = None,
    images: ImageProvider | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FsListTool())
    registry.register(FsReadTool())
    registry.register(FsWriteTool())
    registry.register(FsEditTool())
    registry.register(RunTestsTool(sandbox))
    registry.register(CodeExecTool(sandbox))
    registry.register(FormatTool())
    registry.register(ImageGenerateTool(images or ProceduralProvider()))
    for tool in browser_tools():
        registry.register(tool)
    for tool in desktop_tools(desktop):
        registry.register(tool)
    return registry
