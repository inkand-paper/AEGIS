"""Desktop automation: the backend contract and the data the tools return (Phase 7).

The tools (aegis.tools.desktop) never touch the operating system directly. They talk to a
`DesktopBackend`. The real one is aegis.desktop_windows (Windows UI Automation through pywinauto).
Tests use a simulated desktop. Keeping this line clean is what makes the safety rules testable
without a screen.

How the agent is kept on a leash (see also aegis.apppolicy):
  * Only windows of programs named with --allow-app are visible or usable.
  * Controls are used through their accessibility actions (invoke, select, toggle, set value),
    never by moving the mouse to a screen position, so a window that is covering another one
    cannot receive a click meant for something else.
  * A control is addressed by id AND its exact name; if they do not match the call is refused.
  * Every click states what should be true afterwards and fails if it is not.
  * Clicks on controls with sensitive names (delete, send, pay, install, yes ...) always wait for
    a human, and password-like fields are never typed into.
"""

from typing import Protocol

from pydantic import BaseModel, ConfigDict


class DesktopError(Exception):
    """An action could not be done. The message is shown to the model, so keep it useful."""


class DesktopUnavailable(DesktopError):
    """Desktop automation cannot work here at all (not Windows, library missing)."""


class WindowInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    app: str  # executable name in lower case, for example "notepad.exe"
    foreground: bool = False


class ControlInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    role: str  # button, edit, checkbox, radio, menuitem, listitem, tab, link, combobox, text ...
    name: str
    value: str | None = None  # current text of an edit field; never filled for password fields
    enabled: bool = True
    password: bool = False


class DesktopBackend(Protocol):
    """What the tools need from the operating system. All calls are blocking."""

    def check(self) -> None:
        """Raise DesktopUnavailable (with advice) if this backend cannot work here."""

    def list_windows(self) -> list[WindowInfo]:
        """Every visible top-level window, whatever program owns it."""

    def launch(self, app: str) -> None:
        """Start a program by executable name. The tool has already checked the allowlist."""

    def controls(self, window_id: str) -> list[ControlInfo]:
        """The usable controls of a window. Remembers them so ids stay valid for later calls."""

    def invoke(self, window_id: str, control_id: int) -> None:
        """Press a button, follow a link, pick a menu item, select a tab or toggle a box."""

    def set_text(self, window_id: str, control_id: int, text: str) -> None:
        """Replace the text of an edit field."""

    def screenshot(self, window_id: str) -> bytes:
        """PNG bytes of one window."""
