"""Deterministic verifier (design doc section 9). It is NOT an LLM.

P3: no success claim without evidence. The model saying "I am done" is only a proposal; this
module is the only thing that decides whether a task may become COMPLETED.

Acceptance criteria are plain strings given with `--accept`:

    tool_used:calculator        a tool with that name returned successfully
    result_contains:391         some successful tool result contains this text
    file_read:notes.txt         fs_read successfully read that workspace file
    answer_contains:391         the final answer mentions this text

Richer evidence (TEST_PASS, FILE_DIFF, ...) arrives with the Phase 3 tools.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from aegis.models import Evidence, EvidenceKind, Plan, StepStatus, VerificationResult

VERIFIER_VERSION = "0.1"


class CriterionKind(StrEnum):
    TOOL_USED = "tool_used"
    RESULT_CONTAINS = "result_contains"
    FILE_READ = "file_read"
    ANSWER_CONTAINS = "answer_contains"


@dataclass(frozen=True)
class Criterion:
    kind: CriterionKind
    argument: str
    text: str  # the original string, used as the check name


def parse_criterion(text: str) -> Criterion:
    kind_text, separator, argument = text.partition(":")
    kinds = ", ".join(f"{kind.value}:<value>" for kind in CriterionKind)
    try:
        kind = CriterionKind(kind_text.strip())
    except ValueError:
        raise ValueError(f"unknown acceptance criterion {text!r}. Supported: {kinds}") from None
    argument = argument.strip()
    if not separator or not argument:
        raise ValueError(f"criterion {text!r} needs a value, for example {kind.value}:<value>")
    return Criterion(kind=kind, argument=argument, text=text.strip())


def _normalize_path(path: str) -> str:
    cleaned = path.replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def _check(
    name: str, passed: bool, message: str, evidence_ids: list[str] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "evidence_ids": evidence_ids or [],
        "message": message,
    }


class Verifier:
    def verify(
        self,
        *,
        task_id: str,
        plan: Plan,
        evidence: list[Evidence],
        criteria: list[Criterion],
        final_answer: str | None,
        unmatched_tool_calls: int,
    ) -> VerificationResult:
        tool_evidence = [e for e in evidence if e.kind is EvidenceKind.TOOL_RESULT]
        checks: list[dict[str, Any]] = []

        not_done = [s.index for s in plan.steps if s.status is not StepStatus.DONE]
        checks.append(
            _check(
                "plan_complete",
                not not_done,
                "every step is done" if not not_done else f"steps not done: {not_done}",
            )
        )
        checks.append(
            _check(
                "no_indeterminate_calls",
                unmatched_tool_calls == 0,
                "every tool call has a result"
                if unmatched_tool_calls == 0
                else f"{unmatched_tool_calls} tool call(s) have no recorded result",
            )
        )
        checks.append(
            _check(
                "tool_evidence",
                bool(tool_evidence),
                f"{len(tool_evidence)} successful tool result(s) recorded"
                if tool_evidence
                else "no successful tool call: a task that never used a tool cannot be verified",
                [e.id for e in tool_evidence][:10],
            )
        )
        checks.extend(self._check_criterion(c, tool_evidence, final_answer) for c in criteria)

        return VerificationResult(
            id=uuid.uuid4().hex,
            task_id=task_id,
            passed=all(check["passed"] for check in checks),
            checks=checks,
            created_at=datetime.now(UTC),
            verifier_version=VERIFIER_VERSION,
        )

    def _check_criterion(
        self, criterion: Criterion, evidence: list[Evidence], final_answer: str | None
    ) -> dict[str, Any]:
        name = f"criterion:{criterion.text}"
        argument = criterion.argument

        if criterion.kind is CriterionKind.ANSWER_CONTAINS:
            found = final_answer is not None and argument.lower() in final_answer.lower()
            return _check(
                name,
                found,
                "the final answer contains it" if found else "the final answer does not contain it",
            )

        if criterion.kind is CriterionKind.TOOL_USED:
            matches = [e for e in evidence if e.payload.get("tool_name") == argument]
        elif criterion.kind is CriterionKind.FILE_READ:
            wanted = _normalize_path(argument)
            matches = [
                e
                for e in evidence
                if e.payload.get("tool_name") == "fs_read"
                and _normalize_path(str(e.payload.get("arguments", {}).get("path", ""))) == wanted
            ]
        else:  # RESULT_CONTAINS
            matches = [e for e in evidence if argument in json.dumps(e.payload.get("result"))]

        return _check(
            name,
            bool(matches),
            f"backed by {len(matches)} tool result(s)" if matches else "no matching tool result",
            [e.id for e in matches][:10],
        )
