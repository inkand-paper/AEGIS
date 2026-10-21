from datetime import UTC, datetime
from typing import Any

import pytest

from aegis.models import Evidence, EvidenceKind, Plan, Step, StepStatus
from aegis.verifier import CriterionKind, Verifier, parse_criterion


def make_plan(*statuses: StepStatus) -> Plan:
    steps = [
        Step(id=f"s{n}", plan_id="p", index=n, description=f"step {n}", status=status)
        for n, status in enumerate(statuses, start=1)
    ]
    return Plan(id="p", task_id="t", steps=steps)


def make_evidence(tool_name: str, arguments: dict[str, Any], result: dict[str, Any]) -> Evidence:
    return Evidence(
        id=f"e-{tool_name}-{len(str(result))}",
        task_id="t",
        step_id="s1",
        kind=EvidenceKind.TOOL_RESULT,
        payload={"tool_name": tool_name, "arguments": arguments, "result": result, "outcome": "ok"},
        created_at=datetime.now(UTC),
        source_tool_call_id="c1",
    )


CALC = make_evidence(
    "calculator", {"expression": "17 * 23"}, {"expression": "17 * 23", "value": 391}
)
READ = make_evidence("fs_read", {"path": "notes\\a.txt"}, {"path": "notes\\a.txt", "content": "hi"})


def verify(
    criteria: list[str] | None = None,
    evidence: list[Evidence] | None = None,
    statuses: tuple[StepStatus, ...] = (StepStatus.DONE,),
    answer: str | None = "The answer is 391",
    unmatched: int = 0,
):  # type: ignore[no-untyped-def]
    return Verifier().verify(
        task_id="t",
        plan=make_plan(*statuses),
        evidence=[CALC, READ] if evidence is None else evidence,
        criteria=[parse_criterion(c) for c in criteria or []],
        final_answer=answer,
        unmatched_tool_calls=unmatched,
    )


def failed_names(result) -> set[str]:  # type: ignore[no-untyped-def]
    return {c["name"] for c in result.checks if not c["passed"]}


def test_a_complete_evidenced_task_passes() -> None:
    result = verify()
    assert result.passed is True
    assert {c["name"] for c in result.checks} == {
        "plan_complete",
        "no_indeterminate_calls",
        "tool_evidence",
    }
    assert result.verifier_version


def test_unfinished_steps_fail() -> None:
    result = verify(statuses=(StepStatus.DONE, StepStatus.PENDING))
    assert failed_names(result) == {"plan_complete"}


def test_a_task_without_any_tool_evidence_cannot_pass() -> None:
    result = verify(evidence=[])
    assert result.passed is False
    assert failed_names(result) == {"tool_evidence"}


def test_calls_without_results_fail() -> None:
    assert failed_names(verify(unmatched=1)) == {"no_indeterminate_calls"}


@pytest.mark.parametrize(
    ("criterion", "expected_pass"),
    [
        ("tool_used:calculator", True),
        ("tool_used:fs_list", False),
        ("result_contains:391", True),
        ("result_contains:999", False),
        ("file_read:notes/a.txt", True),  # path separators are normalised
        ("file_read:./notes/a.txt", True),
        ("file_read:other.txt", False),
        ("answer_contains:391", True),
        ("answer_contains:ANSWER", True),  # case-insensitive
        ("answer_contains:banana", False),
    ],
)
def test_criteria(criterion: str, expected_pass: bool) -> None:
    result = verify([criterion])
    check = next(c for c in result.checks if c["name"] == f"criterion:{criterion}")
    assert check["passed"] is expected_pass
    assert result.passed is expected_pass


def test_evidence_backed_criteria_cite_their_evidence() -> None:
    result = verify(["tool_used:calculator"])
    check = next(c for c in result.checks if c["name"].startswith("criterion:"))
    assert check["evidence_ids"] == [CALC.id]


def test_a_claim_in_the_answer_alone_is_never_enough() -> None:
    """P3: the answer mentions 391, but with no tool evidence the task still fails."""
    result = verify(["answer_contains:391"], evidence=[])
    assert result.passed is False


@pytest.mark.parametrize("text", ["bogus:1", "tool_used", "tool_used:", "tool_used:  "])
def test_bad_criteria_are_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="criterion"):
        parse_criterion(text)


def test_criterion_parsing() -> None:
    criterion = parse_criterion(" result_contains: 391 ")
    assert criterion.kind is CriterionKind.RESULT_CONTAINS
    assert criterion.argument == "391"


# ---------------------------------------------------------------- Phase 3 criteria


def kinded(
    kind: EvidenceKind, tool: str, arguments: dict[str, Any], result: dict[str, Any], n: int
) -> Evidence:
    return make_evidence(tool, arguments, result).model_copy(update={"kind": kind, "id": f"e{n}"})


PASSING = {"exit_code": 0, "passed": 3, "failed": 0, "errors": 0, "skipped": 0, "output": ""}
WRITE = {"path": "calc.py", "changed": True, "dry_run": False}


def test_tests_pass_needs_a_passing_run() -> None:
    assert failed_names(verify(["tests_pass"])) == {"criterion:tests_pass"}
    run_ = kinded(EvidenceKind.TEST_PASS, "shell_run_tests", {}, PASSING, 1)
    assert verify(["tests_pass"], evidence=[run_]).passed is True


def test_tests_pass_goes_stale_after_a_later_file_change() -> None:
    run_ = kinded(EvidenceKind.TEST_PASS, "shell_run_tests", {}, PASSING, 1)
    edit = kinded(EvidenceKind.FILE_DIFF, "fs_edit", {"path": "calc.py"}, WRITE, 2)
    stale = verify(["tests_pass"], evidence=[run_, edit])
    assert failed_names(stale) == {"criterion:tests_pass"}
    assert "run the tests again" in next(
        c["message"] for c in stale.checks if c["name"] == "criterion:tests_pass"
    )
    rerun = kinded(EvidenceKind.TEST_PASS, "shell_run_tests", {}, PASSING, 3)
    assert verify(["tests_pass"], evidence=[run_, edit, rerun]).passed is True


def test_a_change_before_the_test_run_is_fine() -> None:
    edit = kinded(EvidenceKind.FILE_DIFF, "fs_edit", {"path": "calc.py"}, WRITE, 1)
    run_ = kinded(EvidenceKind.TEST_PASS, "shell_run_tests", {}, PASSING, 2)
    assert verify(["tests_pass"], evidence=[edit, run_]).passed is True


def test_file_written_matches_only_real_changes_to_that_path() -> None:
    edit = kinded(EvidenceKind.FILE_DIFF, "fs_edit", {"path": "./calc.py"}, WRITE, 1)
    assert verify(["file_written:calc.py"], evidence=[edit]).passed is True
    assert failed_names(verify(["file_written:other.py"], evidence=[edit])) == {
        "criterion:file_written:other.py"
    }
    reading = [READ]  # reading a file is not writing it
    assert verify(["file_written:notes/a.txt"], evidence=reading).passed is False


def test_tests_pass_takes_no_value() -> None:
    assert parse_criterion("tests_pass").kind is CriterionKind.TESTS_PASS
    with pytest.raises(ValueError, match="takes no value"):
        parse_criterion("tests_pass:yes")
