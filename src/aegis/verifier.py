"""Deterministic verifier (design doc section 9). It is NOT an LLM.

P3: no success claim without evidence. The model saying "I am done" is only a proposal; this
module is the only thing that decides whether a task may become COMPLETED.

Acceptance criteria are plain strings given with `--accept`:

    tool_used:calculator        a tool with that name returned successfully
    result_contains:391         some successful tool result contains this text
    file_read:notes.txt         fs_read successfully read that workspace file
    answer_contains:391         the final answer mentions this text
    file_written:src/app.py     fs_write or fs_edit really changed that workspace file
    tests_pass                  shell_run_tests exited 0 with at least one passing test
    page_contains:Thanks        a page the browser really showed contained this text (any
                                page it opened, read or clicked through; case-insensitive)
    url_contains:/done          the browser's LAST page had this in its URL (the end state)
    desktop_contains:Saved      a program window the desktop tools really read contained this text
                                (any window they opened, inspected or clicked in; case-insensitive)
    image_saved:art/logo.png    image_generate saved that file, AND the file on disk right now is
                                still a well-formed image with exactly the recorded SHA-256

Staleness rule (section 9.1): a passing test run only counts if no workspace file changed after
it. "I ran the tests, then edited a file, then claimed success" fails here.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from aegis.images import ImageError, inspect_image
from aegis.models import Evidence, EvidenceKind, Plan, StepStatus, VerificationResult

VERIFIER_VERSION = "0.5"


class CriterionKind(StrEnum):
    TOOL_USED = "tool_used"
    RESULT_CONTAINS = "result_contains"
    FILE_READ = "file_read"
    ANSWER_CONTAINS = "answer_contains"
    FILE_WRITTEN = "file_written"
    TESTS_PASS = "tests_pass"
    PAGE_CONTAINS = "page_contains"
    URL_CONTAINS = "url_contains"
    DESKTOP_CONTAINS = "desktop_contains"
    IMAGE_SAVED = "image_saved"


NO_ARGUMENT = frozenset({CriterionKind.TESTS_PASS})


@dataclass(frozen=True)
class Criterion:
    kind: CriterionKind
    argument: str
    text: str  # the original string, used as the check name


def parse_criterion(text: str) -> Criterion:
    kind_text, separator, argument = text.partition(":")
    kinds = ", ".join(
        kind.value if kind in NO_ARGUMENT else f"{kind.value}:<value>" for kind in CriterionKind
    )
    try:
        kind = CriterionKind(kind_text.strip())
    except ValueError:
        raise ValueError(f"unknown acceptance criterion {text!r}. Supported: {kinds}") from None
    argument = argument.strip()
    if kind in NO_ARGUMENT:
        if argument:
            raise ValueError(f"criterion {kind.value!r} takes no value")
        return Criterion(kind=kind, argument="", text=kind.value)
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
        workspace: Path | None = None,
    ) -> VerificationResult:
        # Every piece of evidence the controller records comes from a successful tool call.
        tool_evidence = [e for e in evidence if e.kind is not EvidenceKind.HUMAN_APPROVAL]
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
        checks.extend(
            self._check_criterion(c, tool_evidence, final_answer, workspace) for c in criteria
        )

        return VerificationResult(
            id=uuid.uuid4().hex,
            task_id=task_id,
            passed=all(check["passed"] for check in checks),
            checks=checks,
            created_at=datetime.now(UTC),
            verifier_version=VERIFIER_VERSION,
        )

    @staticmethod
    def _check_tests_pass(name: str, evidence: list[Evidence]) -> dict[str, Any]:
        # Evidence is in the order it was recorded, so list position tells us what came first.
        last_change = max(
            (i for i, e in enumerate(evidence) if e.kind is EvidenceKind.FILE_DIFF), default=-1
        )
        passes = [(i, e) for i, e in enumerate(evidence) if e.kind is EvidenceKind.TEST_PASS]
        fresh = [e for i, e in passes if i > last_change]
        if fresh:
            return _check(
                name, True, f"{len(fresh)} passing test run(s) after the last file change",
                [e.id for e in fresh][:10],
            )  # fmt: skip
        if passes:
            return _check(
                name,
                False,
                "tests passed earlier, but files changed afterwards: run the tests again",
                [e.id for _, e in passes][:10],
            )
        return _check(name, False, "no test run with exit code 0 and at least one passing test")

    @staticmethod
    def _check_image_saved(
        name: str, argument: str, evidence: list[Evidence], workspace: Path | None
    ) -> dict[str, Any]:
        """The recorded hash must match the file as it is NOW (P3: evidence, not memory)."""
        wanted = _normalize_path(argument)
        saved = [
            e
            for e in evidence
            if e.kind is EvidenceKind.ARTIFACT_HASH
            and e.payload.get("tool_name") == "image_generate"
            and _normalize_path(str((e.payload.get("result") or {}).get("path", ""))) == wanted
        ]
        if not saved:
            return _check(name, False, "image_generate never saved that file")
        latest = saved[-1]  # the last save of this path is the one that counts
        ids = [latest.id]
        if workspace is None:
            return _check(name, False, "no workspace to check the file in", ids)
        root = workspace.resolve()
        target = (root / wanted).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            return _check(name, False, "the file is not in the workspace any more", ids)
        try:
            info = inspect_image(target.read_bytes())
        except (ImageError, OSError) as exc:
            return _check(name, False, f"the file on disk is not a valid image: {exc}", ids)
        recorded = (latest.payload.get("result") or {}).get("sha256")
        if info.sha256 != recorded:
            return _check(name, False, "the file changed after image_generate saved it", ids)
        return _check(
            name, True, f"{info.format} {info.width}x{info.height}, sha256 {info.sha256[:12]}", ids
        )

    def _check_criterion(
        self,
        criterion: Criterion,
        evidence: list[Evidence],
        final_answer: str | None,
        workspace: Path | None = None,
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

        if criterion.kind is CriterionKind.TESTS_PASS:
            return self._check_tests_pass(name, evidence)

        if criterion.kind is CriterionKind.IMAGE_SAVED:
            return self._check_image_saved(name, argument, evidence, workspace)

        if criterion.kind is CriterionKind.FILE_WRITTEN:
            wanted = _normalize_path(argument)
            matches = [
                e
                for e in evidence
                if e.kind is EvidenceKind.FILE_DIFF
                and e.payload.get("tool_name") in {"fs_write", "fs_edit"}
                and _normalize_path(str(e.payload.get("arguments", {}).get("path", ""))) == wanted
            ]
        elif criterion.kind is CriterionKind.PAGE_CONTAINS:
            matches = [
                e
                for e in evidence
                if e.kind is EvidenceKind.BROWSER_STATE
                and argument.lower() in str((e.payload.get("result") or {}).get("text", "")).lower()
            ]
        elif criterion.kind is CriterionKind.DESKTOP_CONTAINS:
            matches = [
                e
                for e in evidence
                if e.kind is EvidenceKind.DESKTOP_STATE
                and argument.lower() in str((e.payload.get("result") or {}).get("text", "")).lower()
            ]
        elif criterion.kind is CriterionKind.URL_CONTAINS:
            states = [e for e in evidence if e.kind is EvidenceKind.BROWSER_STATE]
            last = states[-1:] if states else []
            matches = [
                e for e in last if argument in str((e.payload.get("result") or {}).get("url", ""))
            ]
        elif criterion.kind is CriterionKind.TOOL_USED:
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
