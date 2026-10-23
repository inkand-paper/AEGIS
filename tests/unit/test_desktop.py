"""Phase 7: desktop tools, the app allowlist and the sensitive-action gate, on a simulated desktop.

The real Windows backend (aegis.desktop_windows) is NOT covered here: it needs a screen. It is
checked by hand with scripts/desktop_smoke.py.
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fakedesktop import FakeDesktop
from fakes import ScriptedModel, call_tools, plan_json, say

from aegis.actions import render_actions
from aegis.apppolicy import is_secret_field, is_sensitive, normalise_apps
from aegis.controller import Controller
from aegis.executor import Executor
from aegis.memory import EventLog
from aegis.models import EventType, TaskState, ToolCall, ToolOutcome
from aegis.tools import default_registry
from aegis.tools.base import PolicyError, ToolContext, ToolError
from aegis.tools.desktop import ClickArgs

# ------------------------------------------------------------------ the allowlist (pure)


def test_apps_are_cleaned_and_given_an_exe_suffix() -> None:
    assert normalise_apps(["Notepad", "calc.EXE", "notepad.exe"]) == ["calc.exe", "notepad.exe"]


@pytest.mark.parametrize(
    "bad", ["", "C:\\Windows\\notepad.exe", "/usr/bin/x", "a b.exe", "*.exe", "x" * 70 + ".exe"]
)
def test_paths_and_odd_names_are_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        normalise_apps([bad])


@pytest.mark.parametrize(
    "denied",
    ["cmd", "powershell.exe", "python", "wt", "chrome.exe", "explorer", "regedit", "msedge"],
)
def test_programs_that_bypass_other_controls_cannot_be_allowed(denied: str) -> None:
    with pytest.raises(ValueError, match="cannot be allowed"):
        normalise_apps([denied])


@pytest.mark.parametrize(
    "name",
    ["Delete", "Yes", "Send message", "Pay now", "Install", "Don't save", "Sign in", "Replace",
     "Allow access", "OK to proceed", "Run"],
)  # fmt: skip
def test_sensitive_names_are_recognised(name: str) -> None:
    assert is_sensitive(name)


@pytest.mark.parametrize("name", ["Save", "Cancel", "No", "Open", "Text editor", "Next", "Close"])
def test_ordinary_names_are_not_sensitive(name: str) -> None:
    assert not is_sensitive(name)


def test_secret_fields_are_recognised_by_name() -> None:
    assert is_secret_field("Password") and is_secret_field("Card number")
    assert not is_secret_field("File name")


# ------------------------------------------------------------------ the tools


def registry_with(desk: FakeDesktop) -> Any:
    return default_registry(None, desk)


def ctx(
    tmp_path: Path, apps: list[str] | None = None, effects: set[str] | None = None
) -> ToolContext:
    return ToolContext(
        task_id="t1",
        workspace=tmp_path,
        allowed_effects=frozenset(effects or {"external"}),
        allowed_apps=frozenset(apps if apps is not None else ["notepad.exe"]),
    )


def run_tool(registry: Any, name: str, args: dict[str, Any], c: ToolContext) -> Any:
    tool = registry.get(name)
    return asyncio.run(tool.execute(tool.arg_model.model_validate(args), c))


def test_only_windows_of_allowed_programs_are_visible(tmp_path: Path) -> None:
    desk = FakeDesktop()
    registry = registry_with(desk)
    desk.launch("notepad.exe")
    shown = run_tool(registry, "desktop_windows", {}, ctx(tmp_path)).windows
    assert [(w.app, w.title) for w in shown] == [("notepad.exe", "Untitled - Notepad")]
    with pytest.raises(ToolError, match="does not belong to an allowed program"):
        run_tool(registry, "desktop_inspect", {"window": "w100"}, ctx(tmp_path))  # the calculator


def test_launch_only_starts_allowed_programs(tmp_path: Path) -> None:
    desk = FakeDesktop()
    registry = registry_with(desk)
    with pytest.raises(PolicyError, match="not an allowed program"):
        run_tool(registry, "desktop_launch", {"app": "login.exe"}, ctx(tmp_path))
    assert desk.calls == []  # nothing was started
    state = run_tool(registry, "desktop_launch", {"app": "Notepad.EXE"}, ctx(tmp_path))
    assert state.window.title == "Untitled - Notepad"
    assert [c.name for c in state.controls] == ["Text editor", "Save"]


def test_typing_is_read_back_and_checked(tmp_path: Path) -> None:
    desk = FakeDesktop()
    registry = registry_with(desk)
    run_tool(registry, "desktop_launch", {"app": "notepad.exe"}, ctx(tmp_path))
    state = run_tool(
        registry,
        "desktop_type",
        {"window": "w1", "control": 0, "name": "Text editor", "text": "hello"},
        ctx(tmp_path),
    )
    assert "hello" in state.text and desk.windows["w1"].controls[0].value == "hello"


def test_a_control_must_match_by_id_and_exact_name(tmp_path: Path) -> None:
    desk = FakeDesktop()
    registry = registry_with(desk)
    run_tool(registry, "desktop_launch", {"app": "notepad.exe"}, ctx(tmp_path))
    before = list(desk.calls)
    wrong_name = {"window": "w1", "control": 1, "name": "Text editor", "expect_text": "x"}
    with pytest.raises(ToolError, match="is 'Save', not 'Text editor'"):
        run_tool(registry, "desktop_click", wrong_name, ctx(tmp_path))
    gone = {"window": "w1", "control": 9, "name": "Save", "expect_text": "x"}
    with pytest.raises(ToolError, match="not in that window any more"):
        run_tool(registry, "desktop_click", gone, ctx(tmp_path))
    assert desk.calls == before  # nothing was pressed


def test_the_agent_never_types_into_a_password_field(tmp_path: Path) -> None:
    desk = FakeDesktop()
    registry = registry_with(desk)
    c = ctx(tmp_path, apps=["login.exe"])
    run_tool(registry, "desktop_launch", {"app": "login.exe"}, c)
    with pytest.raises(PolicyError, match="password or secret field"):
        run_tool(
            registry,
            "desktop_type",
            {"window": "w1", "control": 1, "name": "Password", "text": "hunter2"},
            c,
        )
    assert desk.windows["w1"].controls[1].value == ""
    inspected = run_tool(registry, "desktop_inspect", {"window": "w1"}, c)
    assert next(x for x in inspected.controls if x.name == "Password").password is True


def test_a_click_must_state_what_it_expects() -> None:
    with pytest.raises(ValueError, match="expect_text, expect_window_title or"):
        ClickArgs(window="w1", control=0, name="Save")


def test_a_click_whose_post_condition_fails_reports_the_unexpected_dialog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("aegis.tools.desktop.POST_CONDITION_WAIT_S", 0.3)
    desk = FakeDesktop(files={"note.txt": "old"})
    registry = registry_with(desk)
    c = ctx(tmp_path)
    run_tool(registry, "desktop_launch", {"app": "notepad.exe"}, c)
    run_tool(
        registry,
        "desktop_click",
        {"window": "w1", "control": 1, "name": "Save", "expect_window_title": "Save As"},
        c,
    )
    run_tool(
        registry,
        "desktop_type",
        {"window": "w2", "control": 0, "name": "File name", "text": "note.txt"},
        c,
    )
    with pytest.raises(ToolError) as info:
        run_tool(
            registry,
            "desktop_click",
            {"window": "w2", "control": 1, "name": "Save", "expect_window_closed": "Save As"},
            c,
        )
    message = str(info.value)
    assert "unexpected window appeared" in message
    assert "Confirm Save As" in message and "already exists" in message
    assert "'Yes', 'No'" in message
    assert desk.files["note.txt"] == "old"  # the dialog was shown, the file was not touched


def test_screenshots_are_saved_in_the_workspace_with_a_hash(tmp_path: Path) -> None:
    desk = FakeDesktop()
    registry = registry_with(desk)
    run_tool(registry, "desktop_launch", {"app": "notepad.exe"}, ctx(tmp_path))
    first = run_tool(registry, "desktop_screenshot", {"window": "w1"}, ctx(tmp_path))
    second = run_tool(registry, "desktop_screenshot", {"window": "w1"}, ctx(tmp_path))
    assert first.path == "screenshots/shot-001.png" and second.path == "screenshots/shot-002.png"
    import hashlib

    assert hashlib.sha256((tmp_path / first.path).read_bytes()).hexdigest() == first.sha256


# ------------------------------------------------------------------ the dispatcher gates


def call(tool: str, arguments: dict[str, Any]) -> ToolCall:
    from datetime import UTC, datetime

    return ToolCall(
        id="c1", task_id="t1", step_id="s1", tool_name=tool, arguments=arguments,
        requested_at=datetime.now(UTC),
    )  # fmt: skip


def test_desktop_tools_are_refused_when_no_program_was_allowed(tmp_path: Path) -> None:
    executor = Executor(registry_with(FakeDesktop()), EventLog(tmp_path / "e.db"))
    result = asyncio.run(
        executor.execute(call("desktop_windows", {}), ctx(tmp_path, apps=[], effects={"external"}))
    )
    assert result.record.outcome is ToolOutcome.POLICY_VIOLATION
    assert "no program was allowed" in (result.record.error or "")


def test_sensitive_clicks_need_approval_even_when_everything_else_is_pre_approved(
    tmp_path: Path,
) -> None:
    desk = FakeDesktop(files={"note.txt": "old"})
    registry = registry_with(desk)
    run_tool(registry, "desktop_launch", {"app": "notepad.exe"}, ctx(tmp_path))
    desk._add("Dialog", "notepad.exe", [])  # make a window with a Yes button
    window = desk.windows["w2"]
    from fakedesktop import FakeControl

    window.controls.append(FakeControl("button", "Yes"))
    executor = Executor(registry, EventLog(tmp_path / "e.db"))
    everything = ctx(tmp_path, effects={"workspace", "execute", "external"})
    click_yes = call(
        "desktop_click", {"window": "w2", "control": 0, "name": "Yes", "expect_text": "Yes"}
    )
    refused = asyncio.run(executor.execute(click_yes, everything))
    assert refused.record.outcome is ToolOutcome.APPROVAL_REQUIRED
    assert "sensitive" in (refused.record.error or "")
    assert ("invoke", "w2", "Yes") not in desk.calls
    approved = asyncio.run(executor.execute(click_yes, everything, approved=True))
    assert approved.record.outcome is ToolOutcome.OK
    assert ("invoke", "w2", "Yes") in desk.calls


def test_a_sensitive_effect_cannot_be_declared_up_front(tmp_path: Path) -> None:
    controller = Controller(
        ScriptedModel(), default_registry(), EventLog(tmp_path / "e.db"),
        workspaces_dir=tmp_path / "ws",
    )  # fmt: skip
    with pytest.raises(ValueError, match="cannot pre-approve"):
        controller.create_task("x", allowed_effects=["sensitive"])


# ------------------------------------------------------------------ whole workflows


def agent(
    tmp_path: Path, model: ScriptedModel, desk: FakeDesktop, **kwargs: Any
) -> tuple[Controller, EventLog]:
    events = EventLog(tmp_path / "aegis.db")
    controller = Controller(
        model,
        default_registry(None, desk),
        events,
        workspaces_dir=tmp_path / "workspaces",
        max_turns_per_step=12,
        **kwargs,
    )
    return controller, events


def click(window: str, control: int, name: str, **expect: str) -> tuple[str, dict[str, Any]]:
    return ("desktop_click", {"window": window, "control": control, "name": name, **expect})


def typing(window: str, control: int, name: str, text: str) -> tuple[str, dict[str, Any]]:
    return ("desktop_type", {"window": window, "control": control, "name": name, "text": text})


def save_note_script(filename: str = "note.txt") -> list[Any]:
    return [
        call_tools(("desktop_launch", {"app": "notepad.exe"})),
        call_tools(typing("w1", 0, "Text editor", "Meeting at noon")),
        call_tools(click("w1", 1, "Save", expect_window_title="Save As")),
        call_tools(typing("w2", 0, "File name", filename)),
    ]


def test_workflow_write_and_save_a_note_with_a_visible_action_log(tmp_path: Path) -> None:
    """Phase 7 exit criterion: a repeatable desktop workflow completes with a visible log."""
    desk = FakeDesktop()
    model = ScriptedModel(
        say(plan_json("Write the note", "Save it as note.txt")),
        call_tools(("desktop_launch", {"app": "notepad.exe"})),
        call_tools(typing("w1", 0, "Text editor", "Meeting at noon")),
        say("Typed."),
        call_tools(click("w1", 1, "Save", expect_window_title="Save As")),
        call_tools(typing("w2", 0, "File name", "note.txt")),
        call_tools(click("w2", 1, "Save", expect_window_title="note.txt - Notepad")),
        call_tools(("desktop_screenshot", {"window": "w1"})),
        say("Saved note.txt."),
    )
    controller, events = agent(tmp_path, model, desk)
    task = controller.create_task(
        "Write a note and save it",
        acceptance=["desktop_contains:Meeting at noon", "tool_used:desktop_screenshot"],
        allowed_apps=["notepad.exe"],
        allowed_effects=["workspace"],
    )
    finished = asyncio.run(controller.run(task.id))
    assert finished.state is TaskState.COMPLETED, finished.last_error
    assert desk.files == {"note.txt": "Meeting at noon"}
    assert (tmp_path / "workspaces" / task.id / "screenshots" / "shot-001.png").exists()

    log = render_actions(events.events(task.id), only_prefix="desktop_")
    lines = log.splitlines()
    assert [ln.split("(")[0].split()[-1] for ln in lines] == [
        "desktop_launch",
        "desktop_type",
        "desktop_click",
        "desktop_type",
        "desktop_click",
        "desktop_screenshot",
    ]
    assert all("-> ok" in ln for ln in lines)
    assert "name='Save'" in lines[2] and "expect_window_title='Save As'" in lines[2]
    kinds = {e.payload["kind"] for e in events.events(task.id, EventType.EVIDENCE_ADDED)}
    assert {"desktop_state", "artifact_hash"} <= kinds


def test_workflow_an_unexpected_dialog_is_dismissed_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 7 exit criterion (recovery): the surprise dialog is handled, nothing is overwritten."""
    monkeypatch.setattr("aegis.tools.desktop.POST_CONDITION_WAIT_S", 0.3)
    desk = FakeDesktop(files={"note.txt": "precious"})
    model = ScriptedModel(
        say(plan_json("Try to save the note as note.txt")),
        *save_note_script(),
        call_tools(click("w2", 1, "Save", expect_window_closed="Save As")),  # surprise dialog
        call_tools(click("w3", 2, "No", expect_window_closed="Confirm Save As")),  # recover
        call_tools(click("w2", 2, "Cancel", expect_window_closed="Save As")),
        say("note.txt already exists, so I did not overwrite it."),
    )
    controller, events = agent(tmp_path, model, desk)
    task = controller.create_task(
        "Save the note as note.txt",
        acceptance=["desktop_contains:Untitled"],
        allowed_apps=["notepad.exe"],
    )
    finished = asyncio.run(controller.run(task.id))
    assert finished.state is TaskState.COMPLETED, finished.last_error
    assert desk.files == {"note.txt": "precious"}
    outcomes = [e.payload["outcome"] for e in events.events(task.id, EventType.TOOL_RESULT)]
    assert outcomes.count("tool_error") == 1  # the click that met the dialog
    assert "approval_required" not in outcomes
    log = render_actions(events.events(task.id), only_prefix="desktop_")
    assert "Confirm Save As" in log


def test_workflow_an_unexpected_dialog_asking_yes_waits_for_a_human(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 7 exit criterion (approval): 'Yes' to an overwrite prompt is never clicked alone."""
    monkeypatch.setattr("aegis.tools.desktop.POST_CONDITION_WAIT_S", 0.3)
    desk = FakeDesktop(files={"note.txt": "precious"})
    model = ScriptedModel(
        say(plan_json("Save the note as note.txt, replacing the old file")),
        *save_note_script(),
        call_tools(click("w2", 1, "Save", expect_window_closed="Save As")),
        call_tools(click("w3", 1, "Yes", expect_window_closed="Confirm Save As")),
        say("Replaced."),
    )
    controller, events = agent(tmp_path, model, desk)
    task = controller.create_task(
        "Save the note as note.txt",
        acceptance=["desktop_contains:note.txt"],
        allowed_apps=["notepad.exe"],
    )
    paused = asyncio.run(controller.run(task.id))
    assert paused.state is TaskState.WAITING_FOR_APPROVAL
    assert desk.files == {"note.txt": "precious"}  # not replaced yet
    pending = controller.snapshot(task.id)["pending_approval"]
    assert pending["tool_name"] == "desktop_click"
    assert pending["arguments"]["name"] == "Yes"

    finished = asyncio.run(controller.approve(task.id))
    assert finished.state is TaskState.COMPLETED, finished.last_error
    assert desk.files == {"note.txt": "Meeting at noon"}  # replaced only after the human said yes
    log = render_actions(events.events(task.id), only_prefix="desktop_")
    assert "[approved by a human]" in log


def test_workflow_a_password_field_is_never_filled(tmp_path: Path) -> None:
    desk = FakeDesktop()
    model = ScriptedModel(
        say(plan_json("Open the sign-in window and fill in the form")),
        call_tools(("desktop_launch", {"app": "login.exe"})),
        call_tools(typing("w1", 0, "Username", "me")),
        call_tools(typing("w1", 1, "Password", "hunter2")),
        say("I filled in the user name but I cannot type a password."),
    )
    controller, events = agent(tmp_path, model, desk)
    task = controller.create_task(
        "Sign in", acceptance=["desktop_contains:Sign in"], allowed_apps=["login.exe"]
    )
    finished = asyncio.run(controller.run(task.id))
    assert finished.state is TaskState.COMPLETED, finished.last_error
    outcomes = [e.payload["outcome"] for e in events.events(task.id, EventType.TOOL_RESULT)]
    assert outcomes == ["ok", "ok", "policy_violation"]
    assert desk.windows["w1"].controls[1].value == ""
    assert not any(
        "hunter2" in str(c) for c in desk.calls if c[0] == "set_text" and c[2] == "Password"
    )


def test_a_task_without_apps_is_not_offered_desktop_tools(tmp_path: Path) -> None:
    controller, _ = agent(tmp_path, ScriptedModel(), FakeDesktop())
    plain = controller.create_task("x")
    boxed = controller.create_task("x", allowed_apps=["notepad.exe"])
    names = lambda t: {x["name"] for x in controller._tool_listing(t)}  # noqa: E731
    assert not {n for n in names(plain) if n.startswith("desktop_")}
    assert {n for n in names(boxed) if n.startswith("desktop_")} == {
        "desktop_windows",
        "desktop_launch",
        "desktop_inspect",
        "desktop_click",
        "desktop_type",
        "desktop_screenshot",
    }


def test_the_desktop_tools_say_so_when_there_is_no_backend(tmp_path: Path) -> None:
    executor = Executor(default_registry(None, None), EventLog(tmp_path / "e.db"))
    result = asyncio.run(executor.execute(call("desktop_windows", {}), ctx(tmp_path)))
    assert result.record.outcome is ToolOutcome.UNAVAILABLE
    assert "not available here" in (result.record.error or "")
