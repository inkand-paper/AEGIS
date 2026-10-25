"""`aegis bench ...` commands."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from aegis.bench import report as reporting
from aegis.bench.metrics import TaskResult
from aegis.bench.oracle import grade
from aegis.bench.runner import ResumeMismatch, new_run_dir, prepare_workspace, run_suite
from aegis.bench.spec import (
    Suite,
    freeze,
    frozen_digest,
    load_suite,
    select,
    suite_path,
)
from aegis.bench.validate import validate_suite
from aegis.config import get_settings
from aegis.model_client import ModelClient
from aegis.providers import PROVIDERS, ProviderConfigError, build_client

USAGE_ERROR = 4
EXTERNAL_CATEGORIES = {"file_manipulation", "coding", "testing", "safety"}  # no browser, no kill


def add_parser(commands: Any) -> None:
    bench = commands.add_parser("bench", help="run the benchmark suite (Phase 9)")
    sub = bench.add_subparsers(dest="bench_command", required=True)

    def suite_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("suite", type=Path, help="a suite directory or suite.json")

    def selection(p: argparse.ArgumentParser) -> None:
        p.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
        p.add_argument("--only", action="append", default=[], metavar="TASK_ID")
        p.add_argument("--category")
        p.add_argument("--limit", type=int)

    listing = sub.add_parser("list", help="list the tasks")
    suite_arg(listing)
    selection(listing)
    check = sub.add_parser("validate", help="prove every oracle can pass and cannot be skipped")
    suite_arg(check)
    pin = sub.add_parser("freeze", help="pin the suite file by its SHA-256")
    suite_arg(pin)
    run = sub.add_parser("run", help="run tasks with the configured model and grade them")
    suite_arg(run)
    selection(run)
    run.add_argument("--out", type=Path, default=None, help="default: data/bench")
    run.add_argument("--deny-approvals", action="store_true", help="do not approve; leave waiting")
    run.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="continue an earlier run directory: finished tasks are kept, the rest are run",
    )
    run.add_argument("--allow-unfrozen", action="store_true", help="run a suite that is not frozen")
    show = sub.add_parser("show", help="print a results.json")
    show.add_argument("results", type=Path)
    versus = sub.add_parser("compare", help="two results.json side by side")
    versus.add_argument("first", type=Path)
    versus.add_argument("second", type=Path)
    export = sub.add_parser("export", help="lay tasks out so ANOTHER agent can be run on them")
    suite_arg(export)
    selection(export)
    export.add_argument("directory", type=Path)
    grading = sub.add_parser("grade", help="grade workspaces made by another agent")
    suite_arg(grading)
    grading.add_argument("directory", type=Path)


def _load(path: Path) -> tuple[Suite, str] | None:
    try:
        return load_suite(path)
    except (OSError, ValueError) as exc:
        print(f"error: cannot load the suite: {exc}", file=sys.stderr)
        return None


def command(args: argparse.Namespace, model: ModelClient | None = None) -> int:
    name = args.bench_command
    if name in {"show", "compare"}:
        return _show(args)
    if name == "freeze":
        if not suite_path(args.suite).is_file():
            print("error: no such suite", file=sys.stderr)
            return USAGE_ERROR
        try:
            digest = freeze(args.suite)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return USAGE_ERROR
        print(f"frozen: sha256 {digest}")
        return 0
    loaded = _load(args.suite)
    if loaded is None:
        return USAGE_ERROR
    suite, digest = loaded
    if name == "validate":
        problems = validate_suite(suite)
        for task_id, items in problems.items():
            for item in items:
                print(f"{task_id or '(suite)'}: {item}")
        hard = {k: v for k, v in problems.items() if k}
        print(f"{len(suite.tasks)} tasks, {len(hard)} with problems")
        return 1 if hard else 0
    if name == "grade":
        return _grade(suite, digest, args.directory)
    try:
        tasks = select(
            suite, split=args.split, only=args.only, category=args.category, limit=args.limit
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE_ERROR
    if name == "list":
        for t in tasks:
            print(f"{t.id:<26}{t.category:<20}{t.split}")
        print(f"{len(tasks)} tasks")
        return 0
    if name == "export":
        return _export(suite, digest, tasks, args.directory)
    return _run(args, suite, digest, tasks, model)


def _run(
    args: argparse.Namespace,
    suite: Suite,
    digest: str,
    tasks: list[Any],
    model: ModelClient | None,
) -> int:
    pinned = frozen_digest(args.suite)
    if not args.allow_unfrozen:
        if pinned is None:
            print(
                "error: this suite is not frozen, so a run could not be reproduced. Review the "
                "tasks, then `aegis bench freeze` it (or pass --allow-unfrozen).",
                file=sys.stderr,
            )
            return USAGE_ERROR
        if pinned != digest:
            print(
                "error: suite.json changed since it was frozen. Freeze again on purpose, or "
                "pass --allow-unfrozen.",
                file=sys.stderr,
            )
            return USAGE_ERROR
    if not tasks:
        print("error: no tasks selected", file=sys.stderr)
        return USAGE_ERROR
    settings = get_settings()
    if model is None:
        try:
            model = build_client(settings)
        except ProviderConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return USAGE_ERROR
    preset = PROVIDERS.get(settings.provider.strip().lower())
    label = f"{settings.provider}/{settings.model or (preset.model if preset else '?')}"
    if args.split in {"holdout", "all"}:
        print(
            "note: holdout tasks are included. Use them to measure, not to tune.", file=sys.stderr
        )
    out = args.out or (settings.data_dir / "bench")
    client = model

    def progress(r: TaskResult) -> None:
        mark = "ok  " if r.success else "FAIL"
        line = f"{mark} {r.id:<26}{r.state:<22}{r.model_calls} model calls"
        if not r.success:
            line += "  " + reporting.why(r.to_json())[:120]
        print(line, file=sys.stderr)

    run_dir = args.resume or new_run_dir(out)
    print(f"run directory: {run_dir}", file=sys.stderr)
    try:
        results = asyncio.run(
            run_suite(
                suite,
                tasks,
                lambda _task: client,
                out,
                digest=digest,
                model_label=label,
                approve=not args.deny_approvals,
                progress=progress,
                run_dir=run_dir,
            )
        )
    except ResumeMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE_ERROR
    except KeyboardInterrupt:
        print("\ninterrupted. What finished is saved. Continue with:", file=sys.stderr)
        print(f"  aegis bench run {args.suite} --resume {run_dir}", file=sys.stderr)
        print(f"  (results so far: {run_dir / 'results.json'})", file=sys.stderr)
        return 3
    print()
    print(reporting.render(results))
    print(f"\nresults: {results['results_path']}")
    if results.get("stopped"):
        print(
            "\nThe provider stopped serving requests (a rate or daily token limit). What",
            file=sys.stderr,
        )
        print(
            "finished is saved and the unfinished tasks are NOT counted as failures.",
            file=sys.stderr,
        )
        print(
            "Wait for the limit to reset, then continue:\n"
            f"  aegis bench run {args.suite} --resume {run_dir}",
            file=sys.stderr,
        )
        return 3
    return 0


def _show(args: argparse.Namespace) -> int:
    try:
        if args.bench_command == "show":
            print(reporting.render(json.loads(args.results.read_text(encoding="utf-8"))))
        else:
            first = json.loads(args.first.read_text(encoding="utf-8"))
            second = json.loads(args.second.read_text(encoding="utf-8"))
            print(reporting.compare(first, second))
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: cannot read results: {exc}", file=sys.stderr)
        return USAGE_ERROR
    return 0


def _export(suite: Suite, digest: str, tasks: list[Any], directory: Path) -> int:
    chosen = [t for t in tasks if t.category in EXTERNAL_CATEGORIES]
    skipped = [t.id for t in tasks if t.category not in EXTERNAL_CATEGORIES]
    if directory.exists() and any(directory.iterdir()):
        print("error: that directory is not empty", file=sys.stderr)
        return USAGE_ERROR
    for task in chosen:
        root = directory / task.id
        workspace = root / "workspace"
        workspace.mkdir(parents=True)
        prepare_workspace(task, workspace, root)
        (root / "objective.txt").write_text(task.objective + "\n", encoding="utf-8")
    manifest = {"suite_sha256": digest, "tasks": [t.id for t in chosen]}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"exported {len(chosen)} tasks to {directory}")
    if skipped:
        print(f"not exported (browser or kill-and-resume tasks need AEGIS itself): {len(skipped)}")
    print("Run your agent on each <task>/workspace with <task>/objective.txt, then:")
    print("  aegis bench grade <suite> <directory>   (put a final reply in <task>/answer.txt)")
    return 0


def _grade(suite: Suite, digest: str, directory: Path) -> int:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print(
            "error: no manifest.json; was this directory made by `bench export`?", file=sys.stderr
        )
        return USAGE_ERROR
    if manifest.get("suite_sha256") != digest:
        print("error: these workspaces were exported from a different suite file", file=sys.stderr)
        return USAGE_ERROR
    by_id = {t.id: t for t in suite.tasks}
    passed = 0
    for task_id in manifest["tasks"]:
        task = by_id.get(task_id)
        root = directory / task_id
        if task is None or not (root / "workspace").is_dir():
            print(f"skip {task_id}: missing")
            continue
        answer_file = root / "answer.txt"
        answer = answer_file.read_text(encoding="utf-8") if answer_file.is_file() else ""
        canary_file = root / "canary" / "secret.txt"
        canary = (canary_file, task.setup.canary) if task.setup.canary is not None else None
        result = grade(task, root / "workspace", answer, canary)
        passed += result.passed
        why = "" if result.passed else "  " + "; ".join(result.failures)[:110]
        print(f"{'ok  ' if result.passed else 'FAIL'} {task_id:<26}{why}")
    total = len(manifest["tasks"])
    print(f"\n{passed}/{total} passed the oracle")
    return 0
