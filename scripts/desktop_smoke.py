"""Manual check of the real Windows desktop backend (design doc: Phase 7 exit, manual part).

Run on Windows:   uv run python scripts/desktop_smoke.py

It uses NO model and NO agent: it starts Notepad, reads its controls, types a line, reads it back
and saves a screenshot of the window, printing PASS or FAIL for each step. It does not close
Notepad (close it yourself and choose "Don't save"). Paste the whole output if anything fails.
"""

import sys
import time
from pathlib import Path

from aegis.desktop import DesktopError, DesktopUnavailable
from aegis.desktop_windows import WindowsDesktop

APP = "notepad.exe"
LINE = "AEGIS desktop check"
failures = 0


def step(label: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    backend = WindowsDesktop()
    try:
        backend.check()
    except DesktopUnavailable as exc:
        print(f"cannot run: {exc}")
        return 2
    step("backend available", True)

    before = {w.id for w in backend.list_windows()}
    step("listed windows", True, f"{len(before)} visible")
    try:
        backend.launch(APP)
    except DesktopError as exc:
        step("launched notepad", False, str(exc))
        return 1
    window = None
    for _ in range(40):
        time.sleep(0.25)
        fresh = [w for w in backend.list_windows() if w.app == APP and w.id not in before]
        if fresh:
            window = fresh[0]
            break
    step("a Notepad window appeared with program name notepad.exe", window is not None)
    if window is None:
        print("  (If Notepad opened but was not found, the program name of its window differs.)")
        print("  Windows seen:", [(w.app, w.title) for w in backend.list_windows()][:15])
        return 1
    print(f"  window id={window.id} title={window.title!r}")

    controls = backend.controls(window.id)
    step("read controls", bool(controls), f"{len(controls)} controls")
    for c in controls[:25]:
        print(f"  {c.id:>3} {c.role:<9} {c.name!r} value={c.value!r} enabled={c.enabled}")
    editor = next((c for c in controls if c.role in {"document", "edit"}), None)
    step("found the text area", editor is not None)
    if editor is None:
        return 1

    try:
        backend.set_text(window.id, editor.id, LINE)
        time.sleep(0.5)
        after = next(c for c in backend.controls(window.id) if c.role in {"document", "edit"})
        step("typed text reads back", (after.value or "").strip() == LINE, repr(after.value))
    except DesktopError as exc:
        step("typed text reads back", False, str(exc))

    try:
        png = backend.screenshot(window.id)
        out = Path("desktop_smoke.png")
        out.write_bytes(png)
        step("screenshot is a PNG", png.startswith(b"\x89PNG"), f"{len(png)} bytes -> {out}")
    except DesktopError as exc:
        step("screenshot is a PNG", False, str(exc))

    print("\nNotepad is still open: close it and choose Don't save.")
    print("ALL PASSED" if failures == 0 else f"{failures} STEP(S) FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
