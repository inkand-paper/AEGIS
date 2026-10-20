"""Command line entry point (design doc section 11).

    aegis                          prints "AEGIS initialized."
    aegis submit "<objective>"     run a task to a terminal state
    aegis status <task_id>         show where a task stands

Exit codes: 0 completed, 1 failed, 2 waiting for approval, 3 cancelled, 4 usage error.
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn

from aegis.config import get_settings
from aegis.controller import Controller
from aegis.log import setup_logging
from aegis.memory import EventLog
from aegis.model_client import ModelClient
from aegis.models import Budget, TaskState
from aegis.providers import ProviderConfigError, build_client
from aegis.tools import default_registry
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
    submit.add_argument("--max-model-calls", type=int, help="override the model call budget")
    submit.add_argument("--max-tool-calls", type=int, help="override the tool call budget")

    status = commands.add_parser("status", help="show where a task stands")
    status.add_argument("task_id")
    status.add_argument("--json", action="store_true", help="print the raw checkpoint")
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

    events = EventLog(_events_path())
    controller = Controller(
        model,
        default_registry(),
        events,
        workspaces_dir=settings.workspaces_dir,
        default_budget=budget,
    )
    try:
        task = controller.create_task(
            args.objective, acceptance=args.accept, attach=[Path(p) for p in args.attach]
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE_ERROR

    print(f"task {task.id} created. Working...", file=sys.stderr)
    try:
        finished = asyncio.run(controller.run(task.id))
    except KeyboardInterrupt:
        finished = controller.get_task(task.id)
    print()
    print(render(controller.snapshot(task.id)))
    return EXIT_CODES.get(finished.state, 1)


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
    return _status(args)


if __name__ == "__main__":
    raise SystemExit(main())
