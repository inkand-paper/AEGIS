"""A simulated Windows desktop for tests: no screen, but real window and dialog behaviour.

It models a small Notepad: typing text, Save -> a "Save As" dialog, and saving over an existing
file -> a "Confirm Save As" dialog (Yes / No). A second program (calc.exe) is always open and is
never allowed, and login.exe has a password field.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from aegis.desktop import ControlInfo, DesktopError, WindowInfo

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes"


@dataclass
class FakeControl:
    role: str
    name: str
    value: str | None = None
    enabled: bool = True
    password: bool = False
    on_invoke: Callable[[], None] | None = None


@dataclass
class FakeWindow:
    id: str
    title: str
    app: str
    controls: list[FakeControl] = field(default_factory=list)


class FakeDesktop:
    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.files: dict[str, str] = dict(files or {})
        self.windows: dict[str, FakeWindow] = {}
        self.calls: list[tuple[str, ...]] = []  # every call that CHANGED something
        self._next = 1
        self.available = True
        self._add("Calculator", "calc.exe", [FakeControl("button", "Delete")], wid="w100")

    # ---- the backend contract
    def check(self) -> None:
        pass

    def list_windows(self) -> list[WindowInfo]:
        return [
            WindowInfo(id=w.id, title=w.title, app=w.app, foreground=i == len(self.windows) - 1)
            for i, w in enumerate(self.windows.values())
        ]

    def launch(self, app: str) -> None:
        self.calls.append(("launch", app))
        if app == "notepad.exe":
            self._notepad()
        elif app == "login.exe":
            self._add(
                "Sign in",
                app,
                [
                    FakeControl("edit", "Username", ""),
                    FakeControl("edit", "Password", "", password=True),
                ],
            )
        else:
            raise DesktopError(f"{app} was not found")

    def controls(self, window_id: str) -> list[ControlInfo]:
        window = self._window(window_id)
        return [
            ControlInfo(
                id=i,
                role=c.role,
                name=c.name,
                value=None if c.password else c.value,
                enabled=c.enabled,
                password=c.password,
            )
            for i, c in enumerate(window.controls)
        ]

    def invoke(self, window_id: str, control_id: int) -> None:
        control = self._control(window_id, control_id)
        self.calls.append(("invoke", window_id, control.name))
        if control.on_invoke is not None:
            control.on_invoke()

    def set_text(self, window_id: str, control_id: int, text: str) -> None:
        control = self._control(window_id, control_id)
        self.calls.append(("set_text", window_id, control.name, text))
        control.value = text

    def screenshot(self, window_id: str) -> bytes:
        self._window(window_id)
        return PNG

    # ---- helpers
    def _window(self, window_id: str) -> FakeWindow:
        if window_id not in self.windows:
            raise DesktopError(f"window {window_id} no longer exists")
        return self.windows[window_id]

    def _control(self, window_id: str, control_id: int) -> FakeControl:
        window = self._window(window_id)
        if not 0 <= control_id < len(window.controls):
            raise DesktopError("that control is not known any more")
        return window.controls[control_id]

    def _add(
        self, title: str, app: str, controls: list[FakeControl], wid: str | None = None
    ) -> FakeWindow:
        if wid is None:
            wid = f"w{self._next}"
            self._next += 1
        window = FakeWindow(wid, title, app, controls)
        self.windows[wid] = window
        return window

    def _close(self, window: FakeWindow) -> None:
        self.windows.pop(window.id, None)

    def _notepad(self) -> None:
        editor = FakeControl("document", "Text editor", "")
        main = self._add("Untitled - Notepad", "notepad.exe", [editor])
        main.controls.append(
            FakeControl("button", "Save", on_invoke=lambda: self._save_as(main, editor))
        )

    def _save_as(self, main: FakeWindow, editor: FakeControl) -> None:
        name = FakeControl("edit", "File name", "")
        dialog = self._add("Save As", "notepad.exe", [name])

        def save() -> None:
            target = name.value or ""
            if not target:
                return
            if target in self.files:
                self._confirm(main, dialog, editor, target)
            else:
                self._write(main, dialog, editor, target)

        dialog.controls += [
            FakeControl("button", "Save", on_invoke=save),
            FakeControl("button", "Cancel", on_invoke=lambda: self._close(dialog)),
        ]

    def _write(
        self, main: FakeWindow, dialog: FakeWindow, editor: FakeControl, target: str
    ) -> None:
        self.files[target] = editor.value or ""
        main.title = f"{target} - Notepad"
        self._close(dialog)

    def _confirm(
        self, main: FakeWindow, dialog: FakeWindow, editor: FakeControl, target: str
    ) -> None:
        confirm = self._add(
            "Confirm Save As",
            "notepad.exe",
            [FakeControl("text", f"{target} already exists. Do you want to replace it?")],
        )

        def yes() -> None:
            self._close(confirm)
            self._write(main, dialog, editor, target)

        confirm.controls += [
            FakeControl("button", "Yes", on_invoke=yes),
            FakeControl("button", "No", on_invoke=lambda: self._close(confirm)),
        ]


def make(**kwargs: Any) -> FakeDesktop:
    return FakeDesktop(**kwargs)
