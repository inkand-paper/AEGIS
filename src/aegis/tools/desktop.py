"""Desktop tools (design doc section 6, Phase 7). See aegis.desktop for the safety rules.

    desktop_windows     list the windows of the allowed programs
    desktop_launch      start an allowed program
    desktop_inspect     read one window: its controls (with ids) and its text
    desktop_click       press one control, saying what must be true afterwards
    desktop_type        replace the text of an edit field (never a password field)
    desktop_screenshot  save a PNG of one window into the workspace

A click on a control whose name looks sensitive (delete, send, pay, install, yes ...) has the
effect "sensitive", which always waits for a human. If a click makes an unexpected window (a
dialog) appear, the call fails and the error lists the dialog and its buttons, so the agent can
dismiss it or report it instead of clicking on blindly.
"""

import asyncio
import hashlib
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aegis.apppolicy import is_secret_field, is_sensitive
from aegis.desktop import (
    ControlInfo,
    DesktopBackend,
    DesktopError,
    DesktopUnavailable,
    WindowInfo,
)
from aegis.tools.base import (
    PolicyError,
    SideEffects,
    Tool,
    ToolContext,
    ToolError,
    ToolUnavailable,
)

MAX_CONTROLS = 60
MAX_TEXT_CHARS = 2000
MAX_TYPED_CHARS = 500
MAX_SCREENSHOT_BYTES = 5 * 1024 * 1024
POST_CONDITION_WAIT_S = 3.0
LAUNCH_WAIT_S = 10.0
EDITABLE_ROLES = frozenset({"edit", "document", "combobox"})


class DesktopState(BaseModel):
    """What a window shows right now. This is the DESKTOP_STATE evidence."""

    model_config = ConfigDict(extra="forbid")

    window: WindowInfo | None  # None only when no window of the program is open any more
    controls: list[ControlInfo]
    other_windows: list[WindowInfo]  # other windows of the allowed programs, for example dialogs
    new_windows: list[WindowInfo]  # windows that appeared during this call
    note: str | None = None
    text: str  # last on purpose: when a result is cut for the model, the text goes first


class WindowsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    windows: list[WindowInfo]


class Desktop:
    """Shared by the tools: the backend plus the allowlist-checked helpers."""

    def __init__(self, backend: DesktopBackend | None) -> None:
        self._backend = backend

    def backend(self) -> DesktopBackend:
        if self._backend is None:
            raise ToolUnavailable(
                "desktop automation is not available here (it needs Windows and "
                "`uv run aegis desktop check`)"
            )
        return self._backend

    async def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        try:
            return await asyncio.to_thread(fn, *args)
        except DesktopUnavailable as exc:
            raise ToolUnavailable(str(exc)) from None
        except DesktopError as exc:
            raise ToolError(str(exc)) from None

    async def windows(self, ctx: ToolContext) -> list[WindowInfo]:
        every = await self.call(self.backend().list_windows)
        return [w for w in every if w.app in ctx.allowed_apps]

    async def require(self, ctx: ToolContext, window_id: str) -> WindowInfo:
        for window in await self.windows(ctx):
            if window.id == window_id:
                return window
        allowed = ", ".join(sorted(ctx.allowed_apps))
        raise ToolError(
            f"window {window_id!r} does not exist or does not belong to an allowed program "
            f"({allowed}). Call desktop_windows to see the current windows."
        )

    async def find_control(
        self, window_id: str, control_id: int, expected_name: str
    ) -> ControlInfo:
        """The control, only if its id AND exact name match what the model said."""
        controls: list[ControlInfo] = await self.call(self.backend().controls, window_id)
        found = next((c for c in controls if c.id == control_id), None)
        if found is None:
            raise ToolError(
                f"control {control_id} is not in that window any more. "
                "Call desktop_inspect to get the current controls."
            )
        if " ".join(found.name.split()) != " ".join(expected_name.split()):
            raise ToolError(
                f"control {control_id} is {found.name!r}, not {expected_name!r}. "
                "The window changed. Call desktop_inspect to get the current controls."
            )
        if not found.enabled:
            raise ToolError(f"control {control_id} ({found.name!r}) is disabled")
        return found

    async def state(
        self,
        ctx: ToolContext,
        window_id: str,
        *,
        before: set[str] | None = None,
        note: str | None = None,
    ) -> DesktopState:
        windows = await self.windows(ctx)
        target = next((w for w in windows if w.id == window_id), None)
        if target is None and windows:  # the window closed (a dialog was dismissed): show the rest
            target = next((w for w in windows if w.foreground), windows[0])
        controls: list[ControlInfo] = []
        text = ""
        if target is not None:
            controls = (await self.call(self.backend().controls, target.id))[:MAX_CONTROLS]
            parts = [target.title] + [c.name for c in controls if c.name]
            parts += [c.value for c in controls if c.value and not c.password]
            text = " | ".join(" ".join(part.split()) for part in parts)[:MAX_TEXT_CHARS]
        elif before is not None:
            note = "no window of the program is open any more"
        new = [w for w in windows if before is not None and w.id not in before]
        return DesktopState(
            window=target,
            controls=controls,
            other_windows=[w for w in windows if target is None or w.id != target.id],
            new_windows=new,
            note=note,
            text=text,
        )

    async def describe_windows(self, windows: list[WindowInfo]) -> str:
        """One line per window with its buttons: what an unexpected dialog is asking."""
        lines = []
        for window in windows[:3]:
            try:
                controls = await self.call(self.backend().controls, window.id)
            except ToolError:
                controls = []
            buttons = [c.name for c in controls if c.role == "button" and c.name][:6]
            texts = [c.name for c in controls if c.role == "text" and c.name][:2]
            detail = f" says {' / '.join(texts)!r}" if texts else ""
            lines.append(f"{window.title!r}{detail}, buttons: {buttons}")
        return "; ".join(lines)


class _WindowArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window: str = Field(min_length=1, max_length=40, description="window id from desktop_windows")


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LaunchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: str = Field(
        min_length=1, max_length=64, description="an allowed program, e.g. notepad.exe"
    )


class ClickArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window: str = Field(min_length=1, max_length=40)
    control: int = Field(ge=0, description="the control id from the last desktop_inspect")
    name: str = Field(min_length=1, max_length=200, description="that control's exact name")
    expect_text: str | None = Field(
        default=None, min_length=1, max_length=200, description="text the program must show after"
    )
    expect_window_title: str | None = Field(
        default=None, min_length=1, max_length=200, description="a window title that must exist"
    )
    expect_window_closed: str | None = Field(
        default=None, min_length=1, max_length=200, description="a window title that must be gone"
    )

    @model_validator(mode="after")
    def _needs_a_post_condition(self) -> "ClickArgs":
        if not (self.expect_text or self.expect_window_title or self.expect_window_closed):
            raise ValueError(
                "say what must be true after the click: give expect_text, expect_window_title "
                "or expect_window_closed"
            )
        return self


class TypeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window: str = Field(min_length=1, max_length=40)
    control: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=200, description="that control's exact name")
    text: str = Field(max_length=MAX_TYPED_CHARS, description="the new text of the field")


class _Base:
    requires_desktop = True

    def __init__(self, desktop: Desktop) -> None:
        self._desktop = desktop


class DesktopWindowsTool(_Base, Tool[_NoArgs, WindowsResult]):
    name = "desktop_windows"
    description = "List the open windows of the programs this task may use."
    arg_model = _NoArgs
    result_model = WindowsResult
    side_effects = "none"
    requires_network = False
    idempotent = True
    timeout_s = 30.0

    async def execute(self, args: _NoArgs, ctx: ToolContext) -> WindowsResult:
        return WindowsResult(windows=await self._desktop.windows(ctx))


class DesktopInspectTool(_Base, Tool[_WindowArgs, DesktopState]):
    name = "desktop_inspect"
    description = (
        "Read one window: its controls with ids (buttons, fields, menu items) and its text. "
        "Use it after a failed click to see what the window really shows."
    )
    arg_model = _WindowArgs
    result_model = DesktopState
    side_effects = "none"
    requires_network = False
    idempotent = True
    timeout_s = 30.0

    async def execute(self, args: _WindowArgs, ctx: ToolContext) -> DesktopState:
        await self._desktop.require(ctx, args.window)
        return await self._desktop.state(ctx, args.window)


class DesktopLaunchTool(_Base, Tool[LaunchArgs, DesktopState]):
    name = "desktop_launch"
    description = (
        "Start one of the allowed programs and return its window. Only programs the task was "
        "allowed to use can be started."
    )
    arg_model = LaunchArgs
    result_model = DesktopState
    side_effects = "external"
    requires_network = False
    idempotent = False  # a second launch opens a second program
    timeout_s = 45.0

    async def execute(self, args: LaunchArgs, ctx: ToolContext) -> DesktopState:
        app = args.app.strip().lower()
        if app not in ctx.allowed_apps:
            raise PolicyError(
                f"{app!r} is not an allowed program. Allowed: {', '.join(sorted(ctx.allowed_apps))}"
            )
        before = {w.id for w in await self._desktop.windows(ctx)}
        await self._desktop.call(self._desktop.backend().launch, app)
        deadline = time.monotonic() + LAUNCH_WAIT_S
        while True:
            windows = await self._desktop.windows(ctx)
            fresh = [w for w in windows if w.app == app and w.id not in before]
            if fresh:
                return await self._desktop.state(ctx, fresh[0].id, before=before)
            if time.monotonic() >= deadline:
                raise ToolError(f"{app} was started but no window of it appeared")
            await asyncio.sleep(0.3)


class DesktopClickTool(_Base, Tool[ClickArgs, DesktopState]):
    name = "desktop_click"
    description = (
        "Press one control (button, link, menu item, tab, checkbox) of a window, by id plus its "
        "exact name. You MUST say what must be true afterwards (expect_text, expect_window_title "
        "or expect_window_closed). If it is not, the call fails and tells you what the window "
        "shows, including any unexpected dialog. Controls with names like Delete, Send, Pay or "
        "Yes always wait for a human to approve."
    )
    arg_model = ClickArgs
    result_model = DesktopState
    side_effects = "external"
    requires_network = False
    idempotent = False
    repeatable_after_failure = False
    timeout_s = 45.0

    def effective_side_effects(self, args: ClickArgs) -> SideEffects:
        return "sensitive" if is_sensitive(args.name) else "external"

    async def execute(self, args: ClickArgs, ctx: ToolContext) -> DesktopState:
        desk = self._desktop
        await desk.require(ctx, args.window)
        before = {w.id for w in await desk.windows(ctx)}
        control = await desk.find_control(args.window, args.control, args.name)
        # The gate judged the NAME the model gave. It is the real name (checked above), so a
        # model cannot get a sensitive control through by calling it something harmless.
        assert is_sensitive(control.name) == is_sensitive(args.name)
        await desk.call(desk.backend().invoke, args.window, args.control)
        problem = await self._wait_for(ctx, args)
        if problem is not None:
            windows = await desk.windows(ctx)
            fresh = [w for w in windows if w.id not in before]
            extra = ""
            if fresh:
                extra = f" An unexpected window appeared: {await desk.describe_windows(fresh)}."
            raise ToolError(
                f"clicked {args.name!r}, but {problem}.{extra} Call desktop_inspect to see the "
                "window, then dismiss the dialog or do something different, or report that this "
                "cannot be done."
            )
        state = await desk.state(ctx, args.window, before=before)
        if state.new_windows:
            names = ", ".join(repr(w.title) for w in state.new_windows)
            state.note = f"the expected result is there; new window(s) also appeared: {names}"
        else:
            state.note = "the expected result is there"
        return state

    async def _wait_for(self, ctx: ToolContext, args: ClickArgs) -> str | None:
        desk = self._desktop
        deadline = time.monotonic() + POST_CONDITION_WAIT_S
        while True:
            missing: list[str] = []
            windows = await desk.windows(ctx)
            titles = [w.title.lower() for w in windows]
            if args.expect_window_title and not any(
                args.expect_window_title.lower() in t for t in titles
            ):
                missing.append(f"no window titled {args.expect_window_title!r} is open")
            if args.expect_window_closed and any(
                args.expect_window_closed.lower() in t for t in titles
            ):
                missing.append(f"the window {args.expect_window_closed!r} is still open")
            if args.expect_text:
                shown = ""
                for window in windows:
                    for control in await desk.call(desk.backend().controls, window.id):
                        shown += (
                            f" {control.name} {'' if control.password else control.value or ''}"
                        )
                if args.expect_text.lower() not in " ".join(shown.split()).lower():
                    missing.append(f"the text {args.expect_text!r} is not shown")
            if not missing:
                return None
            if time.monotonic() >= deadline:
                return "; ".join(missing)
            await asyncio.sleep(0.2)


class DesktopTypeTool(_Base, Tool[TypeArgs, DesktopState]):
    name = "desktop_type"
    description = (
        "Replace the text of an edit field (by id plus its exact name). The text is set as is: "
        "no key presses or shortcuts. Password and other secret fields are refused. The field is "
        "read back and the call fails if it does not hold the text."
    )
    arg_model = TypeArgs
    result_model = DesktopState
    side_effects = "external"
    requires_network = False
    idempotent = True  # setting the same text twice gives the same field
    repeatable_after_failure = False
    timeout_s = 45.0

    async def execute(self, args: TypeArgs, ctx: ToolContext) -> DesktopState:
        desk = self._desktop
        await desk.require(ctx, args.window)
        control = await desk.find_control(args.window, args.control, args.name)
        if control.password or is_secret_field(control.name):
            raise PolicyError(
                f"{control.name!r} looks like a password or secret field. "
                "The agent never types into those."
            )
        if control.role not in EDITABLE_ROLES:
            raise ToolError(f"{control.name!r} is a {control.role}, not an edit field")
        await desk.call(desk.backend().set_text, args.window, args.control, args.text)
        after = {c.id: c for c in await desk.call(desk.backend().controls, args.window)}
        now = after.get(args.control)
        shown = (now.value if now is not None else None) or ""
        if shown.replace("\r\n", "\n") != args.text.replace("\r\n", "\n"):
            raise ToolError(
                f"typed into {control.name!r}, but the field does not hold the text "
                f"(it shows {shown[:80]!r})"
            )
        state = await desk.state(ctx, args.window)
        state.note = f"{control.name!r} now holds the text"
        return state


class ScreenshotArgs(_WindowArgs):
    pass


class ScreenshotResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str  # inside the workspace
    bytes: int
    sha256: str


class DesktopScreenshotTool(_Base, Tool[ScreenshotArgs, ScreenshotResult]):
    name = "desktop_screenshot"
    description = (
        "Save a PNG picture of one window into the workspace (screenshots/). The picture shows "
        "everything visible in that window, so only use it where that is acceptable."
    )
    arg_model = ScreenshotArgs
    result_model = ScreenshotResult
    side_effects = "workspace"
    requires_network = False
    idempotent = False
    timeout_s = 45.0

    async def execute(self, args: ScreenshotArgs, ctx: ToolContext) -> ScreenshotResult:
        desk = self._desktop
        await desk.require(ctx, args.window)
        png: bytes = await desk.call(desk.backend().screenshot, args.window)
        if not png.startswith(b"\x89PNG"):
            raise ToolError("the screenshot is not a PNG image")
        if len(png) > MAX_SCREENSHOT_BYTES:
            raise ToolError("the screenshot is larger than 5 MB")
        folder = ctx.resolve("screenshots")
        folder.mkdir(exist_ok=True)
        number = len(list(folder.glob("shot-*.png"))) + 1
        target = ctx.resolve(f"screenshots/shot-{number:03d}.png")
        target.write_bytes(png)
        relative = target.relative_to(ctx.workspace.resolve()).as_posix()
        return ScreenshotResult(
            path=relative, bytes=len(png), sha256=hashlib.sha256(png).hexdigest()
        )


def desktop_tools(backend: DesktopBackend | None) -> list[Tool[Any, Any]]:
    desk = Desktop(backend)
    return [
        DesktopWindowsTool(desk),
        DesktopLaunchTool(desk),
        DesktopInspectTool(desk),
        DesktopClickTool(desk),
        DesktopTypeTool(desk),
        DesktopScreenshotTool(desk),
    ]
