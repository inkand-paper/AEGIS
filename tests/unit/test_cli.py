import json
from pathlib import Path

import pytest
from fakes import ScriptedModel, call_tools, plan_json, say

from aegis.config import get_settings
from aegis.main import _submit, build_parser, main


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
