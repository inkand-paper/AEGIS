"""Which desktop applications a task may drive, and which actions are sensitive (design doc 3.2).

A task starts with NO desktop access. `--allow-app notepad.exe` lets it see and use windows that
belong to that program and nothing else. Names are plain executable names, never paths.

Some programs are refused even if you ask for them, because driving them is the same as handing
the agent a way around every other control: terminals and script hosts run any command, browsers
reach any website (the browser tools have a host allowlist for that), installers and system
tools change the machine. The list is best effort. Any program with a built-in terminal, plug-in
system or web view can do the same, so allow only programs you would let a stranger use.
"""

import re
from collections.abc import Iterable

_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}\.exe$")

DENIED_APPS = frozenset(
    """
    cmd powershell pwsh powershell_ise wt windowsterminal conhost bash wsl sh
    python pythonw py pyw node cscript wscript mshta rundll32 regsvr32 msiexec
    regedit regedt32 mmc control taskmgr runas schtasks wmic certutil bitsadmin
    explorer ssh mstsc curl wget ftp telnet
    chrome msedge firefox brave opera vivaldi iexplore
    code devenv idea64 pycharm64
    """.split()
)


def normalise_apps(names: Iterable[str]) -> list[str]:
    """Validate and clean the allowed programs. Raises ValueError with a message for the user."""
    cleaned: list[str] = []
    for raw in names:
        name = str(raw).strip().lower()
        if any(ch in name for ch in "/\\:*? "):
            raise ValueError(f"give a program name such as notepad.exe, not a path: {raw!r}")
        if name and "." not in name:
            name += ".exe"
        if not _NAME.match(name):
            raise ValueError(f"not a program name (expected something like notepad.exe): {raw!r}")
        if name[:-4] in DENIED_APPS:
            raise ValueError(
                f"{name} cannot be allowed: it can run commands, open any website or change the "
                "system, which would bypass the other controls"
            )
        if name not in cleaned:
            cleaned.append(name)
    return sorted(cleaned)


# Names of buttons and menu items that do something hard to undo, spend money, send data out or
# agree to something. Clicking one always needs a human approval, even in an allowed program.
_SENSITIVE = re.compile(
    r"\b(delete|remove|erase|wipe|format|uninstall|install|purchase|buy|pay|checkout|send|"
    r"submit|publish|post|share|confirm|overwrite|replace|discard|reset|clear all|empty|"
    r"do not save|don'?t save|yes|accept|agree|allow|grant|authori[sz]e|proceed|continue|run|"
    r"sign in|log in|login|sign out|log out|unlock|restart|shut ?down|power off)\b",
    re.IGNORECASE,
)


def is_sensitive(name: str) -> bool:
    """True when a control's name suggests an action that must be approved by a human."""
    return bool(_SENSITIVE.search(name))


# Fields that hold secrets. The agent never types into them, whatever the program says they are.
_SECRET_FIELD = re.compile(
    r"\b(pass ?words?|passcode|pin|cvv|cvc|card number|credit card|security code|ssn|"
    r"secret|token|api key|private key|recovery key|otp|one[- ]time)\b",
    re.IGNORECASE,
)


def is_secret_field(name: str) -> bool:
    return bool(_SECRET_FIELD.search(name))
