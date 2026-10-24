"""Checks that a suite is fair before any model is run against it.

For every task: the untouched setup must FAIL the oracle (otherwise the task passes by doing
nothing), and the known-good reference solution must PASS it (otherwise the oracle, or the task,
is broken). This is what keeps a hand- or model-drafted suite honest.
"""

import tempfile
from pathlib import Path

from aegis.bench.oracle import grade
from aegis.bench.runner import prepare_workspace
from aegis.bench.spec import BenchTask, Suite


def _workspace(task: BenchTask, root: Path) -> tuple[Path, tuple[Path, str] | None]:
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    canary = prepare_workspace(task, workspace, root)
    return workspace, canary


def validate_task(task: BenchTask) -> list[str]:
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as scratch:
        base = Path(scratch)
        workspace, canary = _workspace(task, base / "untouched")
        doing_nothing = grade(task, workspace, "", canary)
        if doing_nothing.passed:
            problems.append("the untouched setup already passes the oracle")
        workspace, canary = _workspace(task, base / "solved")
        for name, text in task.reference.files.items():
            target = (workspace / name).resolve()
            if not target.is_relative_to(workspace.resolve()):
                problems.append(f"reference file {name} leaves the workspace")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        solved = grade(task, workspace, task.reference.answer, canary)
        if not solved.passed:
            problems.append(
                "the reference solution fails the oracle: " + "; ".join(solved.failures)
            )
    return problems


def validate_suite(suite: Suite) -> dict[str, list[str]]:
    """Task id -> problems, only for tasks that have any (plus suite-level notes under '')."""
    found: dict[str, list[str]] = {}
    for task in suite.tasks:
        problems = validate_task(task)
        if problems:
            found[task.id] = problems
    holdouts = sum(t.split == "holdout" for t in suite.tasks)
    notes: list[str] = []
    if not 30 <= len(suite.tasks) <= 50:
        notes.append(f"design asks for 30-50 tasks, this suite has {len(suite.tasks)}")
    if holdouts < 10:
        notes.append(f"design asks for at least 10 held-out tasks, this suite has {holdouts}")
    if notes:
        found[""] = notes
    return found
