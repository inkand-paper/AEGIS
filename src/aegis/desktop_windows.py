"""The real desktop backend: Windows UI Automation through pywinauto (Phase 7).

THIS FILE HAS ONLY BEEN CHECKED ON LINUX (imports, the pure helpers, the "not Windows" path).
Its calls into pywinauto have never run against a real screen by the author. Run the manual check
before trusting it:  uv run python scripts/desktop_smoke.py

What it does and does not do:
  * Controls are used through their accessibility patterns (Invoke, Toggle, SelectionItem, Value),
    never by moving the mouse to a screen position. There is no `click_input`.
  * Windows are found by process: the program name of a window is the image name of its process.
    Packaged (UWP) apps are shown by Windows under `ApplicationFrameHost.exe`, so for them the
    program name is wrong and they cannot be allowed by their own name. Use classic programs.
  * Controls with no accessible name are skipped: the agent could not address them safely anyway.
"""

import ctypes
import io
import os
import shutil
import subprocess
import sys
from typing import Any

from aegis.desktop import ControlInfo, DesktopError, DesktopUnavailable, WindowInfo

MAX_CONTROLS_SCANNED = 400
MAX_CONTROLS_KEPT = 120

# UI Automation control type -> the role names the tools use.
ROLE_BY_TYPE = {
    "Button": "button",
    "SplitButton": "button",
    "Edit": "edit",
    "Document": "document",
    "CheckBox": "checkbox",
    "RadioButton": "radio",
    "MenuItem": "menuitem",
    "ListItem": "listitem",
    "TabItem": "tab",
    "Hyperlink": "link",
    "ComboBox": "combobox",
    "Text": "text",
}
INVOKABLE = frozenset({"button", "menuitem", "link"})
SELECTABLE = frozenset({"tab", "listitem", "radio"})


def role_for(control_type: str) -> str | None:
    """The tool role of a UI Automation control type, or None for types the agent ignores."""
    return ROLE_BY_TYPE.get(control_type)


def exe_name(path: str) -> str:
    """'C:\\Windows\\System32\\Notepad.EXE' -> 'notepad.exe'."""
    return os.path.basename(path.replace("\\", "/")).lower()


def _image_name(pid: int) -> str:
    """The executable name of a process, using only ctypes (no extra dependency)."""
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ""
    try:
        size = ctypes.c_ulong(1024)
        buffer = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return exe_name(buffer.value)
        return ""
    finally:
        kernel32.CloseHandle(handle)


class WindowsDesktop:
    """Implements aegis.desktop.DesktopBackend on Windows."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[int, Any]] = {}  # window id -> control id -> wrapper

    # ---------------------------------------------------------------- setup

    def check(self) -> None:
        if sys.platform != "win32":
            raise DesktopUnavailable("desktop automation only works on Windows")
        try:
            import pywinauto  # noqa: F401
        except ImportError:
            raise DesktopUnavailable(
                "the pywinauto package is not installed. Run: uv sync"
            ) from None

    def _desktop(self) -> Any:
        self.check()
        from pywinauto import Desktop

        return Desktop(backend="uia")

    def _window(self, window_id: str) -> Any:
        try:
            handle = int(window_id)
        except ValueError:
            raise DesktopError(f"bad window id {window_id!r}") from None
        window = self._desktop().window(handle=handle)
        if not window.exists():
            raise DesktopError(f"window {window_id} no longer exists")
        return window

    # ---------------------------------------------------------------- reading

    def list_windows(self) -> list[WindowInfo]:
        self.check()
        foreground = ctypes.windll.user32.GetForegroundWindow()  # type: ignore[attr-defined,unused-ignore]
        found: list[WindowInfo] = []
        for window in self._desktop().windows():
            try:
                if not window.is_visible():
                    continue
                title = window.window_text() or ""
                app = _image_name(window.process_id())
                handle = window.handle
            except Exception:  # a window can vanish while we look at it
                continue
            if not app or not title:
                continue
            found.append(
                WindowInfo(id=str(handle), title=title, app=app, foreground=handle == foreground)
            )
        return found

    def controls(self, window_id: str) -> list[ControlInfo]:
        window = self._window(window_id)
        cache: dict[int, Any] = {}
        result: list[ControlInfo] = []
        for wrapper in window.descendants()[:MAX_CONTROLS_SCANNED]:
            if len(result) >= MAX_CONTROLS_KEPT:
                break
            try:
                info = wrapper.element_info
                role = role_for(info.control_type)
                name = (info.name or "").strip()
                if role is None or not wrapper.is_visible():
                    continue
                password = bool(getattr(info, "password", False) or _is_password(wrapper))
                if not name and role not in {"edit", "document"}:
                    continue
                value: str | None = None
                if role in {"edit", "document", "combobox"} and not password:
                    value = _read_value(wrapper)
                control_id = len(result)
                cache[control_id] = wrapper
                result.append(
                    ControlInfo(
                        id=control_id,
                        role=role,
                        name=name or role,
                        value=value,
                        enabled=bool(wrapper.is_enabled()),
                        password=password,
                    )
                )
            except Exception:
                continue
        self._cache[window_id] = cache
        return result

    # ---------------------------------------------------------------- acting

    def _wrapper(self, window_id: str, control_id: int) -> Any:
        wrapper = self._cache.get(window_id, {}).get(control_id)
        if wrapper is None:
            raise DesktopError("that control is not known any more; inspect the window again")
        return wrapper

    def invoke(self, window_id: str, control_id: int) -> None:
        wrapper = self._wrapper(window_id, control_id)
        role = role_for(wrapper.element_info.control_type)
        try:
            if role == "checkbox":
                wrapper.iface_toggle.Toggle()
            elif role in SELECTABLE:
                try:
                    wrapper.iface_selection_item.Select()
                except Exception:
                    wrapper.iface_invoke.Invoke()
            elif role in INVOKABLE:
                wrapper.iface_invoke.Invoke()
            else:
                raise DesktopError(f"a {role} cannot be pressed")
        except DesktopError:
            raise
        except Exception as exc:
            raise DesktopError(
                f"the program did not accept the action ({type(exc).__name__})"
            ) from None

    def set_text(self, window_id: str, control_id: int, text: str) -> None:
        wrapper = self._wrapper(window_id, control_id)
        try:
            wrapper.iface_value.SetValue(text)
            return
        except Exception:
            pass
        try:
            wrapper.set_edit_text(text)
        except Exception as exc:
            raise DesktopError(
                f"the field did not accept the text ({type(exc).__name__})"
            ) from None

    def screenshot(self, window_id: str) -> bytes:
        window = self._window(window_id)
        try:
            image = window.capture_as_image()
        except Exception as exc:
            raise DesktopError(f"could not capture the window ({type(exc).__name__})") from None
        if image is None:
            raise DesktopError("the window is minimized or has nothing to capture")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def launch(self, app: str) -> None:
        self.check()
        path = shutil.which(app)
        if path is None:
            raise DesktopError(f"{app} was not found on this computer's PATH")
        subprocess.Popen(  # noqa: S603 - one allowlisted program, no shell, no arguments
            [path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )


def _is_password(wrapper: Any) -> bool:
    try:
        return bool(wrapper.element_info.element.CurrentIsPassword)
    except Exception:
        return False


def _read_value(wrapper: Any) -> str | None:
    try:
        return str(wrapper.iface_value.CurrentValue)
    except Exception:
        pass
    try:
        text = wrapper.window_text()
        return str(text) if text else ""
    except Exception:
        return None
