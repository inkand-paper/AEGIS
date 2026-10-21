"""Phase 4 exit criterion: kill a real process mid-task, restart, `resume`.

The task is cut off while a tool is running. After `resume` it must reach COMPLETED (the tool
was safe to run again) or WAITING_FOR_APPROVAL (it may have half-run). It must never silently
repeat a side effect.
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unit"))

from crashtools import registry  # noqa: E402
from fakes import ScriptedModel, say  # noqa: E402

from aegis.controller import Controller  # noqa: E402
from aegis.memory import EventLog  # noqa: E402
from aegis.models import EventType, TaskState  # noqa: E402

WORKER = Path(__file__).resolve().parent / "worker.py"


def start_and_kill(data_dir: Path, tool_name: str) -> str:
    """Run the worker until its tool has started, then SIGKILL it. Returns the task id."""
    process = subprocess.Popen(
        [sys.executable, str(WORKER), str(data_dir), tool_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert process.stdout is not None
        task_id = process.stdout.readline().strip()
        assert task_id, "the worker did not report a task id"
        trace = data_dir / "workspaces" / task_id / "trace.txt"
        deadline = time.monotonic() + 30
        while not trace.exists():  # the tool writes its trace the moment it really starts
            assert process.poll() is None, "the worker exited before reaching the tool"
            assert time.monotonic() < deadline, "the worker never reached the tool"
            time.sleep(0.05)
    finally:
        process.kill()  # SIGKILL on POSIX, TerminateProcess on Windows: no cleanup runs
        process.wait(timeout=10)
    return task_id


def restart(data_dir: Path) -> tuple[Controller, EventLog]:
    log = EventLog(data_dir / "aegis.db")
    controller = Controller(
        ScriptedModel(say("Done. The step is finished.")),
        registry(),
        log,
        workspaces_dir=data_dir / "workspaces",
        lease_ttl_s=1.0,
    )
    return controller, log


def trace_lines(data_dir: Path, task_id: str) -> list[str]:
    path = data_dir / "workspaces" / task_id / "trace.txt"
    return path.read_text(encoding="utf-8").splitlines()


def test_a_killed_task_that_may_have_half_run_waits_instead_of_repeating(tmp_path: Path) -> None:
    task_id = start_and_kill(tmp_path, "charge_card")
    assert len(trace_lines(tmp_path, task_id)) == 1  # the charge started once, then the kill

    controller, log = restart(tmp_path)
    assert log.unmatched_tool_calls(task_id)  # the kill left a call with no result
    time.sleep(1.5)  # the dead worker's lease must run out before anyone else may take over

    resumed = asyncio.run(controller.resume(task_id))

    assert resumed.state is TaskState.WAITING_FOR_APPROVAL
    assert len(trace_lines(tmp_path, task_id)) == 1  # NOT charged a second time
    assert log.unmatched_tool_calls(task_id) == []  # the cut-off call was closed on the record
    requested = [e.payload for e in log.events(task_id, EventType.APPROVAL_REQUESTED)]
    assert requested[-1]["tool_name"] == "charge_card"


def test_a_killed_task_whose_call_is_safe_to_repeat_completes(tmp_path: Path) -> None:
    task_id = start_and_kill(tmp_path, "lookup_price")
    controller, log = restart(tmp_path)
    time.sleep(1.5)

    resumed = asyncio.run(controller.resume(task_id))

    assert resumed.state is TaskState.COMPLETED, resumed.last_error
    # The lookup ran again (it is idempotent), and the task did not wait for anybody.
    assert len(trace_lines(tmp_path, task_id)) == 2
    assert log.unmatched_tool_calls(task_id) == []


def test_resuming_right_after_the_kill_is_refused_until_the_lease_expires(tmp_path: Path) -> None:
    from aegis.memory import LeaseHeld

    task_id = start_and_kill(tmp_path, "charge_card")
    controller, _ = restart(tmp_path)
    with pytest.raises(LeaseHeld):
        asyncio.run(controller.resume(task_id))
