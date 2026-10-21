import json
from pathlib import Path

import pytest
from fakes import ScriptedModel, call_tools, plan_json, say

from aegis.config import get_settings
from aegis.main import _approve, _resume, _submit, build_parser, main


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run in an empty folder: no real .env, and data/ and workspaces/ land in tmp_path."""
    monkeypatch.chdir(tmp_path)
    for name in ("AEGIS_API_KEY", "AEGIS_PROVIDER", "AEGIS_MODEL", "AEGIS_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def happy_model() -> ScriptedModel:
    return ScriptedModel(
        say(plan_json("Multiply")),
        call_tools(("calculator", {"expression": "17 * 23"})),
        say("The product is 391."),
    )


def submit(*argv: str, model: ScriptedModel | None = None) -> int:
    return _submit(build_parser().parse_args(["submit", *argv]), model=model)


def test_no_arguments_prints_the_initialized_message(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "AEGIS initialized." in capsys.readouterr().out


def test_submit_runs_a_task_and_prints_a_report(capsys: pytest.CaptureFixture[str]) -> None:
    code = submit("Multiply 17 by 23", "--accept", "result_contains:391", model=happy_model())
    out = capsys.readouterr().out
    assert code == 0
    assert "state:      completed" in out
    assert "verification: PASSED" in out
    assert "[pass] criterion:result_contains:391" in out
    assert "The product is 391." in out


def test_a_failed_task_exits_with_code_1(capsys: pytest.CaptureFixture[str]) -> None:
    code = submit("Multiply", "--accept", "result_contains:999", model=happy_model())
    assert code == 1
    assert "VerificationFailed" in capsys.readouterr().out


def test_budget_flags_override_the_defaults(capsys: pytest.CaptureFixture[str]) -> None:
    model = ScriptedModel(
        say(plan_json("Loop")),
        *[call_tools(("calculator", {"expression": "1 + 1"})) for _ in range(5)],
    )
    code = submit("Loop", "--max-tool-calls", "2", model=model)
    assert code == 1
    assert "BudgetExceeded: tool_calls 2/2" in capsys.readouterr().out


def test_status_shows_a_finished_task(capsys: pytest.CaptureFixture[str]) -> None:
    submit("Multiply", model=happy_model())
    lines = capsys.readouterr().out.splitlines()
    task_id = next(line for line in lines if line.startswith("task:")).split()[1]

    assert main(["status", task_id]) == 0
    assert "state:      completed" in capsys.readouterr().out

    assert main(["status", task_id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["task"]["id"] == task_id


def test_status_of_an_unknown_task_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", "does-not-exist"]) == 4
    assert "unknown task" in capsys.readouterr().err


def test_a_bad_criterion_is_a_usage_error_and_costs_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = ScriptedModel()  # would raise if called
    assert submit("x", "--accept", "bogus", model=model) == 4
    assert "criterion" in capsys.readouterr().err
    assert model.calls == []


def test_a_missing_api_key_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert submit("x") == 4
    assert "AEGIS_API_KEY" in capsys.readouterr().err


def test_attach_copies_files_into_the_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("remember the milk", encoding="utf-8")
    model = ScriptedModel(
        say(plan_json("Read")),
        call_tools(("fs_read", {"path": "notes.txt"})),
        say("It says remember the milk."),
    )
    code = submit(
        "Read notes.txt",
        "--attach",
        str(notes),
        "--accept",
        "file_read:notes.txt",
        model=model,
    )
    assert code == 0
    assert "[pass] criterion:file_read:notes.txt" in capsys.readouterr().out


def test_bad_command_line_arguments_exit_with_code_4() -> None:
    with pytest.raises(SystemExit) as info:
        main(["submit"])  # the objective is missing
    assert info.value.code == 4


# ---------------------------------------------------------------- Phase 3: approval from the CLI


def write_model() -> ScriptedModel:
    return ScriptedModel(
        say(plan_json("Write the file")),
        call_tools(("fs_write", {"path": "hello.txt", "content": "hi\n"})),
    )


def task_id_from(out: str) -> str:
    return next(line for line in out.splitlines() if line.startswith("task:")).split()[1]


def approve_args(task_id: str, *extra: str) -> object:
    return build_parser().parse_args(["approve", task_id, *extra])


def test_a_write_pauses_with_exit_code_2_and_says_how_to_continue(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = submit("Write hello.txt", model=write_model())
    out = capsys.readouterr().out
    assert code == 2
    assert "state:      waiting_for_approval" in out
    assert "waiting for approval:" in out
    assert "tool: fs_write" in out
    assert "content: (1 lines)" in out and "| hi" in out  # the human sees what would be written
    assert f"aegis approve {task_id_from(out)}" in out


def test_allow_writes_skips_the_pause(capsys: pytest.CaptureFixture[str]) -> None:
    model = ScriptedModel(
        say(plan_json("Write the file")),
        call_tools(("fs_write", {"path": "hello.txt", "content": "hi\n"})),
        say("written"),
    )
    assert submit("Write hello.txt", "--allow-writes", model=model) == 0


def test_approve_with_yes_finishes_the_task(capsys: pytest.CaptureFixture[str]) -> None:
    submit("Write hello.txt", "--accept", "file_written:hello.txt", model=write_model())
    task_id = task_id_from(capsys.readouterr().out)

    code = _approve(
        approve_args(task_id, "--yes"),
        model=ScriptedModel(say("written")),  # type: ignore[arg-type]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "state:      completed" in out
    assert "[pass] criterion:file_written:hello.txt" in out


def test_answering_no_runs_nothing(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    submit("Write hello.txt", model=write_model())
    task_id = task_id_from(capsys.readouterr().out)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")

    model = ScriptedModel()  # must not be called
    code = _approve(approve_args(task_id), model=model)  # type: ignore[arg-type]

    assert code == 2
    assert "Nothing was run" in capsys.readouterr().out
    assert model.calls == []
    assert main(["status", task_id]) == 0
    assert "waiting_for_approval" in capsys.readouterr().out


def test_without_a_terminal_approve_demands_yes(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    submit("Write hello.txt", model=write_model())
    task_id = task_id_from(capsys.readouterr().out)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert _approve(approve_args(task_id), model=ScriptedModel()) == 4  # type: ignore[arg-type]
    assert "--yes" in capsys.readouterr().err


def test_approve_rejects_unknown_and_finished_tasks(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["approve", "nope", "--yes"]) == 4
    assert "unknown task" in capsys.readouterr().err
    submit("Multiply", model=happy_model())
    task_id = task_id_from(capsys.readouterr().out)
    assert main(["approve", task_id, "--yes"]) == 4
    assert "nothing is waiting" in capsys.readouterr().err


def test_cancel_denies_a_pending_approval(capsys: pytest.CaptureFixture[str]) -> None:
    submit("Write hello.txt", model=write_model())
    task_id = task_id_from(capsys.readouterr().out)

    assert main(["cancel", task_id]) == 3
    assert "state:      cancelled" in capsys.readouterr().out
    assert main(["approve", task_id, "--yes"]) == 4


def test_cancel_of_an_unknown_task_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["cancel", "nope"]) == 4
    assert "unknown task" in capsys.readouterr().err


def test_the_allow_flags_are_recorded_on_the_task(capsys: pytest.CaptureFixture[str]) -> None:
    submit("Multiply", "--allow-writes", "--allow-exec", model=happy_model())
    task_id = task_id_from(capsys.readouterr().out)
    assert main(["status", task_id, "--json"]) == 0
    task = json.loads(capsys.readouterr().out)["task"]
    assert task["allowed_effects"] == ["execute", "workspace"]


# ---------------------------------------------------------------- Phase 4: resume and report


class ProcessDied(BaseException):
    """Stands in for a killed process: nothing in the controller may catch it."""


def crashing_model() -> ScriptedModel:
    def die(messages: object) -> object:
        raise ProcessDied  # the process is killed while waiting for the model

    return ScriptedModel(
        say(plan_json("Multiply")), call_tools(("calculator", {"expression": "17 * 23"})), die
    )


def crashed_task(capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(ProcessDied):
        _submit(build_parser().parse_args(["submit", "Multiply"]), model=crashing_model())
    capsys.readouterr()
    from aegis.main import _events_path
    from aegis.memory import EventLog

    log = EventLog(_events_path())
    with log._conn:  # find the task id the way a user would: from the database
        row = log._conn.execute("SELECT task_id FROM checkpoints").fetchone()
    return str(row["task_id"])


def resume_args(task_id: str) -> object:
    return build_parser().parse_args(["resume", task_id])


def test_resume_finishes_a_task_whose_process_died(capsys: pytest.CaptureFixture[str]) -> None:
    task_id = crashed_task(capsys)
    code = _resume(resume_args(task_id), model=ScriptedModel(say("The product is 391.")))  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert code == 0
    assert "state:      completed" in out
    assert "recovery (the last time this task was resumed):" in out
    assert "step resumed: 1" in out


def test_resume_refuses_a_finished_task_and_prints_its_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    submit("Multiply", model=happy_model())
    task_id = task_id_from(capsys.readouterr().out)
    assert main(["resume", task_id]) == 4
    captured = capsys.readouterr()
    assert "already completed" in captured.err
    assert "state:      completed" in captured.out


def test_resume_of_an_unknown_task_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["resume", "nope"]) == 4
    assert "unknown task" in capsys.readouterr().err


def test_resume_refuses_while_another_process_holds_the_task(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from aegis.main import _events_path
    from aegis.memory import EventLog

    task_id = crashed_task(capsys)
    EventLog(_events_path()).acquire_lease(task_id, "pid4242-abc", ttl_s=300)
    code = _resume(resume_args(task_id), model=ScriptedModel())  # type: ignore[arg-type]
    assert code == 4
    assert "pid4242-abc" in capsys.readouterr().err


def test_report_lists_the_files_with_hashes_and_replays_the_log(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = ScriptedModel(
        say(plan_json("Write")),
        call_tools(("fs_write", {"path": "out/result.txt", "content": "391\n"})),
        say("done"),
    )
    submit("Write it", "--allow-writes", model=model)
    task_id = task_id_from(capsys.readouterr().out)

    assert main(["report", task_id, "--replay"]) == 0
    out = capsys.readouterr().out
    assert "files in the workspace:" in out
    assert "out/result.txt  4 bytes  sha256:" in out
    assert "replay: OK" in out


def test_report_without_replay_does_not_replay(capsys: pytest.CaptureFixture[str]) -> None:
    submit("Multiply", model=happy_model())
    task_id = task_id_from(capsys.readouterr().out)
    assert main(["report", task_id]) == 0
    assert "replay:" not in capsys.readouterr().out


def test_report_replay_exits_1_for_an_inconsistent_log(capsys: pytest.CaptureFixture[str]) -> None:
    from aegis.main import _events_path
    from aegis.memory import EventLog
    from aegis.models import EventType

    submit("Multiply", model=happy_model())
    task_id = task_id_from(capsys.readouterr().out)
    EventLog(_events_path()).append(task_id, EventType.ERROR, {"note": "after the end"})
    assert main(["report", task_id, "--replay"]) == 1
    out = capsys.readouterr().out
    assert "PROBLEM" in out and "replay: INCONSISTENT" in out


def test_report_of_an_unknown_task_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["report", "nope"]) == 4
