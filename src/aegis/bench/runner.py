"""Runs benchmark tasks through the real controller and grades them with the oracle.

Each task gets its own directory, event log and workspace, so a benchmark run never touches
`data/aegis.db` or `workspaces/`, and runs do not influence each other.
"""

import asyncio
import contextlib
import http.server
import json
import logging
import shutil
import socketserver
import subprocess
import threading
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from aegis.bench.metrics import TaskResult, compute, holdout_gap, provider_failure
from aegis.bench.oracle import grade
from aegis.bench.spec import BenchTask, Suite
from aegis.controller import Controller
from aegis.memory import EventLog
from aegis.model_client import ModelClient
from aegis.models import Budget, EventType, TaskState
from aegis.tools import default_registry
from aegis.verifier import VERIFIER_VERSION

logger = logging.getLogger(__name__)

MAX_APPROVALS = 10  # a task needing more human decisions than this is stopped and left waiting
TERMINAL_STATES = {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}

ModelFactory = Callable[[BenchTask], ModelClient]


class SimulatedInterrupt(BaseException):
    """Stands in for the process being killed. Nothing in the controller may catch it."""


class InterruptingModel:
    """Lets `after` model calls through, then 'kills the process' on the next one, once."""

    def __init__(self, inner: ModelClient, after: int) -> None:
        self._inner = inner
        self._after = after
        self.calls = 0
        self.fired = False

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        if not self.fired and self.calls >= self._after:
            self.fired = True
            raise SimulatedInterrupt
        self.calls += 1
        return await self._inner.generate(*args, **kwargs)


@dataclass
class Site:
    url: str
    host: str


@contextlib.contextmanager
def serve_site(files: dict[str, str], directory: Path) -> Iterator[Site | None]:
    """Serve the task's static pages on 127.0.0.1 for the duration of the task."""
    if not files:
        yield None
        return
    placeholder = "{site_url}"

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    directory.mkdir(parents=True, exist_ok=True)
    server = socketserver.ThreadingTCPServer(
        ("127.0.0.1", 0), partial(Quiet, directory=str(directory))
    )
    server.daemon_threads = True
    host = f"127.0.0.1:{server.server_address[1]}"
    url = f"http://{host}/"
    for name, text in files.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.replace(placeholder, url.rstrip("/")), encoding="utf-8")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Site(url=url, host=host)
    finally:
        server.shutdown()
        server.server_close()


def prepare_workspace(task: BenchTask, workspace: Path, root: Path) -> tuple[Path, str] | None:
    """Write the task's setup files; plant the canary outside the workspace."""
    for name, text in task.setup.files.items():
        target = (workspace / name).resolve()
        if not target.is_relative_to(workspace.resolve()):
            raise ValueError(f"{task.id}: setup file {name} leaves the workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    if task.setup.canary is None:
        return None
    canary = root / "canary" / "secret.txt"
    canary.parent.mkdir(parents=True, exist_ok=True)
    canary.write_text(task.setup.canary, encoding="utf-8")
    return canary, task.setup.canary


def _budget(task: BenchTask) -> Budget:
    return Budget(
        max_model_calls=task.budget.max_model_calls,
        max_tool_calls=task.budget.max_tool_calls,
        max_replans=task.budget.max_replans,
    )


def _controller(model: ModelClient, events: EventLog, root: Path, task: BenchTask) -> Controller:
    return Controller(
        model,
        default_registry(),
        events,
        workspaces_dir=root / "workspaces",
        default_budget=_budget(task),
    )


async def _settle(controller: Controller, task_id: str, approve: bool, counter: list[int]) -> None:
    """Approve what a human would be asked about, counting each approval as an intervention."""
    while controller.get_task(task_id).state is TaskState.WAITING_FOR_APPROVAL:
        if not approve or counter[0] >= MAX_APPROVALS:
            return
        counter[0] += 1
        await controller.approve(task_id)


async def run_task(
    task: BenchTask,
    model_factory: ModelFactory,
    run_dir: Path,
    *,
    approve: bool = True,
) -> TaskResult:
    root = run_dir / task.id
    if root.exists():  # an earlier, unfinished attempt of this task: start clean
        shutil.rmtree(root)
    root.mkdir(parents=True)
    events = EventLog(root / "aegis.db")
    started = datetime.now(UTC)
    result = TaskResult(
        id=task.id, category=task.category, split=task.split, state="error", oracle_passed=False
    )
    try:
        with serve_site(task.setup.site, root / "site") as site:
            objective = task.objective
            hosts = list(task.allow_hosts)
            if site is not None:
                objective = objective.replace("{site_url}", site.url)
                hosts = [h.replace("{site_host}", site.host) for h in hosts]
            real = model_factory(task)
            interrupting = (
                InterruptingModel(real, task.interrupt_after_model_calls)
                if task.interrupt_after_model_calls
                else None
            )
            controller = _controller(interrupting or real, events, root, task)
            effects = [
                e
                for e, on in (("workspace", task.allow_writes), ("execute", task.allow_exec))
                if on
            ]
            created = controller.create_task(
                objective,
                budget=_budget(task),
                acceptance=task.accept,
                allowed_effects=effects,
                allowed_hosts=hosts,
            )
            canary = prepare_workspace(task, created.workspace_path, root)
            counter = [0]
            try:
                await controller.run(created.id)
            except SimulatedInterrupt:
                result.interrupted = True
            if result.interrupted:
                # A fresh controller, as a fresh process would be, resumes from the event log.
                controller = _controller(real, events, root, task)
                await controller.resume(created.id)
            await _settle(controller, created.id, approve, counter)
            final = controller.get_task(created.id)
            snapshot = controller.snapshot(created.id)
            result.recovered = result.interrupted and final.state in TERMINAL_STATES
            result.state = final.state.value
            result.interventions = counter[0]
            usage = events.recompute_usage(created.id)
            result.model_calls = usage.model_calls
            result.tool_calls = usage.tool_calls
            result.tokens = usage.tokens_total
            result.dollars = str(usage.dollars)
            result.wall_s = round((datetime.now(UTC) - started).total_seconds(), 2)
            for event in events.events(created.id):
                if event.type is EventType.TOOL_RESULT:
                    if event.payload.get("outcome") == "policy_violation":
                        result.policy_rejections += 1
                elif event.type is EventType.PLAN_REVISED:
                    result.replans += 1
            if final.last_error:
                result.error = final.last_error
            graded = grade(task, created.workspace_path, snapshot.get("final_answer") or "", canary)
            result.oracle_passed = graded.passed
            result.oracle_failures = graded.failures
            if any(f.startswith("canary_intact") for f in graded.failures):
                result.breaches += 1
    except Exception as exc:  # one broken task must not stop the suite
        logger.exception("task %s crashed the harness", task.id)
        result.error = f"harness error: {type(exc).__name__}: {exc}"
    finally:
        events.close()
    return result


def environment(suite: Suite, digest: str, model_label: str) -> dict[str, Any]:
    commit = None
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        commit = (
            subprocess.run(  # noqa: S603, S607 - fixed argv, no shell
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            or None
        )
    return {
        "suite": suite.name,
        "suite_version": suite.version,
        "suite_sha256": digest,
        "model": model_label,
        "verifier_version": VERIFIER_VERSION,
        "git_commit": commit,
    }


def new_run_dir(out_dir: Path) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    path = out_dir / run_id
    path.mkdir(parents=True)
    return path


class ResumeMismatch(Exception):
    """The run being continued was made with another suite file or another model."""


def _report(
    run_dir: Path,
    suite: Suite,
    digest: str,
    model_label: str,
    approve: bool,
    selected: list[str],
    results: list[TaskResult],
    *,
    complete: bool,
    stopped: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_dir.name,
        "complete": complete,
        "stopped": stopped,
        "selected": selected,
        "written_at": datetime.now(UTC).isoformat(),
        "environment": environment(suite, digest, model_label),
        "approve_policy": "auto" if approve else "deny",
        "metrics": compute(results),
        "holdout_gap": holdout_gap(results),
        "tasks": [r.to_json() for r in results],
    }


def _save(run_dir: Path, report: dict[str, Any]) -> None:
    """Write through a temporary file so a kill never leaves a half-written results.json."""
    target = run_dir / "results.json"
    temporary = run_dir / "results.json.tmp"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(target)


async def run_suite(
    suite: Suite,
    tasks: list[BenchTask],
    model_factory: ModelFactory,
    out_dir: Path,
    *,
    digest: str,
    model_label: str,
    approve: bool = True,
    progress: Callable[[TaskResult], None] | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Run tasks one after another. `results.json` is rewritten after EVERY task, so a Ctrl+C,
    a daily rate limit or a crash keeps everything that finished. Pass an existing `run_dir`
    to continue it: tasks already in its results.json are skipped."""
    results: list[TaskResult] = []
    if run_dir is None:
        run_dir = new_run_dir(out_dir)
    elif (run_dir / "results.json").is_file():
        previous = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        env = previous["environment"]
        if env["suite_sha256"] != digest or env["model"] != model_label:
            raise ResumeMismatch(
                "that run used a different suite file or model, so its results cannot be mixed "
                "with new ones"
            )
        # A task the provider refused to serve was never really attempted: run it again.
        results = [
            r for r in (TaskResult(**row) for row in previous["tasks"]) if not provider_failure(r)
        ]
    selected = [t.id for t in tasks]
    done = {r.id for r in results}
    for task in tasks:
        if task.id in done:
            continue
        result = await run_task(task, model_factory, run_dir, approve=approve)
        if provider_failure(result):
            # Every later task would hit the same wall. Stop, keep what finished, and say why.
            stopped = f"{task.id}: {(result.error or '')[:300]}"
            report = _report(
                run_dir, suite, digest, model_label, approve, selected, results,
                complete=False, stopped=stopped,
            )  # fmt: skip
            _save(run_dir, report)
            report["results_path"] = str(run_dir / "results.json")
            report["run_dir"] = str(run_dir)
            return report
        results.append(result)
        _save(
            run_dir,
            _report(
                run_dir, suite, digest, model_label, approve, selected, results, complete=False
            ),
        )
        if progress:
            progress(result)
    report = _report(run_dir, suite, digest, model_label, approve, selected, results, complete=True)
    _save(run_dir, report)
    report["results_path"] = str(run_dir / "results.json")
    report["run_dir"] = str(run_dir)
    return report


def run(coro: Any) -> Any:
    return asyncio.run(coro)
