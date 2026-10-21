"""Command line entry point (design doc section 11).

    aegis                          prints "AEGIS initialized."
    aegis submit "<objective>"     run a task until it finishes or needs approval
    aegis status <task_id>         show where a task stands
    aegis approve <task_id>        review the pending call, and if you agree, carry on
    aegis cancel <task_id>         stop a task (this also denies a pending approval)
    aegis resume <task_id>         carry on a task whose process died
    aegis report <task_id>         final report and the files the task produced
                                   (--replay also checks the event log for consistency)

Exit codes: 0 completed, 1 failed, 2 waiting for approval, 3 cancelled, 4 usage error.
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import Any, NoReturn

from aegis.config import get_settings
from aegis.controller import TERMINAL, Controller, InvalidTransition, TaskNotFound
from aegis.log import setup_logging
from aegis.memory import EventLog, LeaseHeld
from aegis.model_client import ModelClient
from aegis.models import Budget, Task, TaskState
from aegis.providers import ProviderConfigError, build_client
from aegis.replay import replay
from aegis.tools import default_registry
from aegis.tools.diffing import sha256_hex
from aegis.verifier import parse_criterion

EXIT_CODES = {
    TaskState.COMPLETED: 0,
    TaskState.FAILED: 1,
    TaskState.WAITING_FOR_APPROVAL: 2,
    TaskState.CANCELLED: 3,
}
USAGE_ERROR = 4


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f"error: {message}", file=sys.stderr)
        raise SystemExit(USAGE_ERROR)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="aegis", description="AEGIS: a long-running AI agent.")
    commands = parser.add_subparsers(dest="command")

    submit = commands.add_parser("submit", help="run a task to a terminal state")
    submit.add_argument("objective", help="what you want done, in plain language")
    submit.add_argument(
        "--accept",
        action="append",
        default=[],
        metavar="CRITERION",
        help="acceptance criterion, for example result_contains:391 (repeatable)",
    )
    submit.add_argument(
        "--attach",
        action="append",
        default=[],
        metavar="FILE",
        help="copy a file into the task workspace (repeatable)",
    )
    submit.add_argument(
        "--allow-writes",
        action="store_true",
        help="pre-approve creating and editing files inside the task workspace",
    )
    submit.add_argument(
        "--allow-exec",
        action="store_true",
        help="pre-approve running the project's tests (this RUNS code the model wrote)",
    )
    submit.add_argument("--max-model-calls", type=int, help="override the model call budget")
    submit.add_argument("--max-tool-calls", type=int, help="override the tool call budget")

    status = commands.add_parser("status", help="show where a task stands")
    status.add_argument("task_id")
    status.add_argument("--json", action="store_true", help="print the raw checkpoint")

    approve = commands.add_parser("approve", help="grant the pending approval of a task")
    approve.add_argument("task_id")
    approve.add_argument(
        "--yes", action="store_true", help="skip the question (for scripts; you still own the risk)"
    )

    resume = commands.add_parser("resume", help="carry on a task whose process was interrupted")
    resume.add_argument("task_id")

    report = commands.add_parser("report", help="the final report and the task's files")
    report.add_argument("task_id")
    report.add_argument(
        "--replay",
        action="store_true",
        help="replay the event log and check it; exits 1 if anything is inconsistent",
    )

    cancel = commands.add_parser("cancel", help="cancel a task; a pending approval is denied")
    cancel.add_argument("task_id")
    return parser


def render(snapshot: dict[str, Any]) -> str:
    """A human-readable summary of a checkpoint. Plain ASCII on purpose."""
    task = snapshot["task"]
    usage = task["usage"]
    lines = [
        f"task:       {task['id']}",
        f"state:      {task['state']}",
        f"objective:  {task['objective']}",
        f"workspace:  {task['workspace_path']}",
    ]
    plan = snapshot.get("plan")
    if plan:
        lines.append("plan:")
        for step in plan["steps"]:
            lines.append(f"  {step['index']}. [{step['status']}] {step['description']}")
    verification = snapshot.get("verification")
    if verification:
        lines.append("verification: " + ("PASSED" if verification["passed"] else "FAILED"))
        for check in verification["checks"]:
            mark = "pass" if check["passed"] else "FAIL"
            lines.append(f"  [{mark}] {check['name']}: {check['message']}")
    pending = snapshot.get("pending_approval")
    if pending:
        lines.append("waiting for approval:")
        lines.extend(f"  {line}" for line in describe_pending(pending))
        task_id = task["id"]
        lines.append(
            f"  -> `aegis approve {task_id}` to review it, `aegis cancel {task_id}` to deny"
        )
    recovery = snapshot.get("recovery")
    if recovery:
        lines.append("recovery (the last time this task was resumed):")
        lines.append(f"  step resumed: {recovery.get('resumed_step')}")
        for call in recovery.get("interrupted_calls", []):
            lines.append(f"  cut off before finishing: {call['tool_name']}")
        for name, values in recovery.get("usage_drift", {}).items():
            lines.append(
                f"  usage drift in {name}: recorded {values['recorded']}, "
                f"events say {values['from_events']} (corrected)"
            )
    if snapshot.get("final_answer"):
        lines.append("final answer:")
        lines.extend(f"  {line}" for line in str(snapshot["final_answer"]).splitlines())
    if task.get("last_error"):
        lines.append(f"error:      {task['last_error']}")
    lines.append(
        "usage:      "
        f"model_calls={usage['model_calls']} tool_calls={usage['tool_calls']} "
        f"tokens={usage['tokens_total']} dollars={usage['dollars']} "
        f"wall={float(usage['wall_clock_s']):.1f}s"
    )
    return "\n".join(lines)


def describe_pending(pending: dict[str, Any]) -> list[str]:
    """What a human needs to see before saying yes. Long text is shortened, never hidden."""
    lines = [f"tool: {pending['tool_name']}", f"why:  {pending['reason']}"]
    for name, value in pending["arguments"].items():
        text = value if isinstance(value, str) else json.dumps(value)
        if "\n" in text or len(text) > 100:
            shown = text.splitlines()
            lines.append(f"{name}: ({len(shown)} lines)")
            lines.extend(f"    | {line[:120]}" for line in shown[:40])
            if len(shown) > 40:
                lines.append(f"    | ... {len(shown) - 40} more lines")
        else:
            lines.append(f"{name}: {text}")
    return lines


def _events_path() -> Path:
    return get_settings().data_dir / "aegis.db"


def _submit(args: argparse.Namespace, model: ModelClient | None = None) -> int:
    try:
        for text in args.accept:
            parse_criterion(text)  # validate before spending any model calls
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE_ERROR

    settings = get_settings()
    if model is None:
        try:
            model = build_client(settings)
        except ProviderConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return USAGE_ERROR

    overrides = {
        "max_model_calls": args.max_model_calls,
        "max_tool_calls": args.max_tool_calls,
    }
    try:
        budget = Budget.model_validate(
            {**settings.default_budget.model_dump(), **{k: v for k, v in overrides.items() if v}}
        )
    except ValueError as exc:
        print(f"error: invalid budget: {exc}", file=sys.stderr)
        return USAGE_ERROR

    controller = _controller(model, budget)
    allowed = [
        effect
        for flag, effect in ((args.allow_writes, "workspace"), (args.allow_exec, "execute"))
        if flag
    ]
    try:
        task = controller.create_task(
            args.objective,
            acceptance=args.accept,
            attach=[Path(p) for p in args.attach],
            allowed_effects=allowed,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE_ERROR

    print(f"task {task.id} created. Working...", file=sys.stderr)
    return _drive(controller, task.id, controller.run(task.id))


def _controller(model: ModelClient, budget: Budget | None = None) -> Controller:
    settings = get_settings()
    return Controller(
        model,
        default_registry(),
        EventLog(_events_path()),
        workspaces_dir=settings.workspaces_dir,
        default_budget=budget,
    )


def _drive(controller: Controller, task_id: str, work: Coroutine[Any, Any, Task]) -> int:
    try:
        finished = asyncio.run(work)
    except KeyboardInterrupt:
        finished = controller.get_task(task_id)
    except LeaseHeld as exc:
        print(f"error: {exc}. Wait for it to finish, or for its lease to expire.", file=sys.stderr)
        return USAGE_ERROR
    except InvalidTransition as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE_ERROR
    print()
    print(render(controller.snapshot(task_id)))
    return EXIT_CODES.get(finished.state, 1)


def _approve(args: argparse.Namespace, model: ModelClient | None = None) -> int:
    snapshot = EventLog(_events_path()).load_checkpoint(args.task_id)
    if snapshot is None:
        print(f"error: unknown task {args.task_id}", file=sys.stderr)
        return USAGE_ERROR
    pending = snapshot.get("pending_approval")
    state = snapshot["task"]["state"]
    if state != TaskState.WAITING_FOR_APPROVAL.value or not pending:
        print(f"error: task is {state}; nothing is waiting for approval", file=sys.stderr)
        return USAGE_ERROR

    print(f"Task {args.task_id} wants to run:")
    print("\n".join(f"  {line}" for line in describe_pending(pending)))
    print(f"Workspace (look at the files before you agree): {snapshot['task']['workspace_path']}")
    if not args.yes:
        if not sys.stdin.isatty():
            print("error: not interactive; use --yes to approve unasked", file=sys.stderr)
            return USAGE_ERROR
        if input("Approve this call? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Not approved. Nothing was run. Use `aegis cancel` to stop the task.")
            return EXIT_CODES[TaskState.WAITING_FOR_APPROVAL]

    if model is None:
        try:
            model = build_client(get_settings())
        except ProviderConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return USAGE_ERROR
    controller = _controller(model)
    return _drive(controller, args.task_id, controller.approve(args.task_id))


def _resume(args: argparse.Namespace, model: ModelClient | None = None) -> int:
    snapshot = EventLog(_events_path()).load_checkpoint(args.task_id)
    if snapshot is None:
        print(f"error: unknown task {args.task_id}", file=sys.stderr)
        return USAGE_ERROR
    state = snapshot["task"]["state"]
    if TaskState(state) in TERMINAL:
        print(f"error: task is already {state}; nothing to resume. Its report:", file=sys.stderr)
        print(render(snapshot))
        return USAGE_ERROR
    if model is None:
        try:
            model = build_client(get_settings())
        except ProviderConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return USAGE_ERROR
    print(f"Resuming task {args.task_id} (was {state})...", file=sys.stderr)
    controller = _controller(model)
    return _drive(controller, args.task_id, controller.resume(args.task_id))


def _report(args: argparse.Namespace) -> int:
    log = EventLog(_events_path())
    snapshot = log.load_checkpoint(args.task_id)
    if snapshot is None:
        print(f"error: unknown task {args.task_id}", file=sys.stderr)
        return USAGE_ERROR
    print(render(snapshot))
    files = workspace_files(Path(snapshot["task"]["workspace_path"]))
    print("files in the workspace:" if files else "files in the workspace: (none)")
    for name, size, digest in files:
        print(f"  {name}  {size} bytes  sha256:{digest[:16]}")
    if not args.replay:
        return 0
    result = replay(log, args.task_id)
    print()
    print(
        f"replay: {result.event_count} events, state {result.state.value if result.state else '?'}"
    )
    for note in result.notes:
        print(f"  note: {note}")
    for problem in result.problems:
        print(f"  PROBLEM: {problem}")
    print("replay: OK" if result.ok else "replay: INCONSISTENT")
    return 0 if result.ok else 1


def workspace_files(root: Path, limit: int = 200) -> list[tuple[str, int, str]]:
    """The files a task left behind: (relative path, size, sha256). Symlinks are not followed."""
    found: list[tuple[str, int, str]] = []
    if not root.is_dir():
        return found
    for item in sorted(root.rglob("*")):
        parts = item.relative_to(root).parts
        if item.is_symlink() or not item.is_file() or "__pycache__" in parts:
            continue
        if ".pytest_cache" in parts or len(found) >= limit:
            continue
        data = item.read_bytes()
        found.append((item.relative_to(root).as_posix(), len(data), sha256_hex(data)))
    return found


def _cancel(args: argparse.Namespace) -> int:
    controller = _controller(_NoModel())
    try:
        state = controller.get_task(args.task_id).state
    except TaskNotFound:
        print(f"error: unknown task {args.task_id}", file=sys.stderr)
        return USAGE_ERROR
    if state in {TaskState.RUNNING, TaskState.VERIFYING}:
        print(
            "error: that task is running in another process. Press Ctrl+C in its window "
            "(cancelling a running task from here arrives with Phase 4).",
            file=sys.stderr,
        )
        return USAGE_ERROR
    controller.cancel(args.task_id)
    print(render(controller.snapshot(args.task_id)))
    return EXIT_CODES.get(controller.get_task(args.task_id).state, 1)


class _NoModel:
    """Cancelling never calls the model, so no API key should be needed for it."""

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("cancel does not call the model")


def _status(args: argparse.Namespace) -> int:
    snapshot = EventLog(_events_path()).load_checkpoint(args.task_id)
    if snapshot is None:
        print(f"error: unknown task {args.task_id}", file=sys.stderr)
        return USAGE_ERROR
    print(json.dumps(snapshot, indent=2) if args.json else render(snapshot))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        setup_logging()
        get_settings()  # fails fast if the environment is misconfigured
        print("AEGIS initialized.")
        return 0
    setup_logging()
    if args.command == "submit":
        return _submit(args)
    if args.command == "approve":
        return _approve(args)
    if args.command == "resume":
        return _resume(args)
    if args.command == "report":
        return _report(args)
    if args.command == "cancel":
        return _cancel(args)
    return _status(args)


if __name__ == "__main__":
    raise SystemExit(main())
