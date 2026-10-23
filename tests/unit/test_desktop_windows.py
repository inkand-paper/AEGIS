"""The parts of the real Windows backend that can be checked without Windows.

Everything that talks to pywinauto is checked by hand: uv run python scripts/desktop_smoke.py
"""

import sys

import pytest

from aegis.desktop import DesktopUnavailable
from aegis.desktop_windows import ROLE_BY_TYPE, WindowsDesktop, exe_name, role_for


def test_program_names_come_from_the_image_path() -> None:
    assert exe_name("C:\\Windows\\System32\\Notepad.EXE") == "notepad.exe"
    assert exe_name("D:/apps/Tool.exe") == "tool.exe"
    assert exe_name("") == ""


def test_known_control_types_map_to_roles_and_the_rest_are_ignored() -> None:
    assert role_for("Button") == "button" and role_for("Edit") == "edit"
    assert role_for("TabItem") == "tab" and role_for("Hyperlink") == "link"
    assert role_for("Pane") is None and role_for("TitleBar") is None
    assert set(ROLE_BY_TYPE.values()) >= {"button", "edit", "document", "checkbox", "radio"}


@pytest.mark.skipif(sys.platform == "win32", reason="checks the not-Windows path")
def test_the_backend_says_plainly_that_it_needs_windows() -> None:
    with pytest.raises(DesktopUnavailable, match="only works on Windows"):
        WindowsDesktop().check()
    with pytest.raises(DesktopUnavailable):
        WindowsDesktop().list_windows()
