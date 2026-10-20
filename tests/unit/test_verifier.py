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
READ = make_evidence(
    "fs_read", {"path": "notes\\a.txt"}, {"path": "notes\\a.txt", "content": "hi"}
)


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
