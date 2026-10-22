"""Event-log replay (design doc section 8.3 and the Phase 4 acceptance check).

The events are the source of truth. `replay` walks them in order, rebuilds what the state MUST be
(task state, step statuses, usage, pending approval) and checks the log against the rules the
system promises: legal state changes, one result per call, evidence from real results, a passing
verification before COMPLETED. It then compares the rebuilt state with the checkpoint row.

A checkpoint that disagrees with the log is a problem for a finished task. For a task that is
still open it is expected after a crash, and `aegis resume` repairs it, so it is only a note.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from aegis.controller import _TRANSITIONS, TERMINAL
from aegis.memory import EventLog
from aegis.models import BudgetUsage, EventType, TaskState


@dataclass
class ReplayReport:
    ok: bool
    event_count: int
    state: TaskState | None
    usage: BudgetUsage
    step_statuses: dict[str, str]
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def replay(log: EventLog, task_id: str) -> ReplayReport:
    events = log.events(task_id)
    problems: list[str] = []
    notes: list[str] = []
    state = TaskState.CREATED
    steps: dict[str, str] = {}
    calls: dict[str, int] = {}
    results: dict[str, bool] = {}  # call id -> was it a success
    requested: dict[str, int] = {}  # approvals asked for and not yet granted
    verification_passed = False
    terminal_at: int | None = None
    current_plan: str | None = None
    replans = 0

    if not events:
        return ReplayReport(
            False, 0, None, BudgetUsage(), {}, ["no events recorded for this task id"], []
        )
    if events[0].type is not EventType.TASK_CREATED:
        problems.append(f"the first event is {events[0].type.value}, not task_created")

    for event in events:
        payload = event.payload
        kind = event.type
        if terminal_at is not None:
            problems.append(f"event {event.id} ({kind.value}) came after the task ended")
            terminal_at = None  # report this once, not for every later event

        if kind is EventType.STATE_CHANGED and payload.get("scope") == "task":
            before, after = payload["from"], payload["to"]
            if before != state.value:
                problems.append(
                    f"event {event.id}: moved from {before} but the task was {state.value}"
                )
            try:
                target = TaskState(after)
            except ValueError:
                problems.append(f"event {event.id}: unknown state {after!r}")
                continue
            if target not in _TRANSITIONS[state]:
                problems.append(f"event {event.id}: illegal change {state.value} -> {after}")
            if target is TaskState.COMPLETED and not verification_passed:
                problems.append(f"event {event.id}: COMPLETED without a passing verification")
            state = target
            if state in TERMINAL:
                terminal_at = event.id
        elif kind is EventType.STATE_CHANGED and payload.get("scope") == "step":
            steps[payload["step_id"]] = payload["status"]
        elif kind is EventType.TOOL_CALL:
            call_id = event.correlation_id or ""
            calls[call_id] = calls.get(call_id, 0) + 1
        elif kind is EventType.TOOL_RESULT:
            call_id = event.correlation_id or ""
            if call_id not in calls:
                problems.append(f"event {event.id}: a result for a call that was never recorded")
            if call_id in results:
                problems.append(f"event {event.id}: a second result for call {call_id}")
            results[call_id] = bool(payload.get("success"))
        elif kind is EventType.EVIDENCE_ADDED:
            source = payload.get("source_tool_call_id", "")
            if not results.get(source, False):
                problems.append(f"event {event.id}: evidence from a call that did not succeed")
        elif kind is EventType.PLAN_CREATED:
            current_plan = payload["plan"]["id"]
        elif kind is EventType.PLAN_REVISED:
            replans += 1
            if payload.get("supersedes") != current_plan:
                problems.append(
                    f"event {event.id}: the revised plan does not replace the current one"
                )
            current_plan = payload["plan"]["id"]
            limit = events[0].payload.get("budget", {}).get("max_replans")
            if limit is not None and replans > int(limit):
                problems.append(f"event {event.id}: replan {replans} is over the budget of {limit}")
        elif kind is EventType.VERIFICATION:
            verification_passed = bool(payload.get("passed"))
        elif kind is EventType.APPROVAL_REQUESTED:
            requested[event.correlation_id or ""] = event.id
        elif kind is EventType.APPROVAL_GRANTED:
            if requested.pop(event.correlation_id or "", None) is None:
                problems.append(f"event {event.id}: approval granted that was never requested")

    open_calls = [c for c in calls if c not in results]
    if open_calls:
        message = f"{len(open_calls)} tool call(s) have no result"
        if state in TERMINAL:
            problems.append(message)
        else:
            notes.append(f"{message}: they were cut off, and `aegis resume` will close them")
    duplicated = [c for c, n in calls.items() if n > 1]
    if duplicated:
        problems.append(f"{len(duplicated)} tool call id(s) were recorded more than once")

    usage = log.recompute_usage(task_id)
    problems_or_notes = problems if state in TERMINAL else notes
    snapshot = log.load_checkpoint(task_id)
    if snapshot is None:
        problems.append("there is no checkpoint for this task")
    else:
        recorded = snapshot["task"]
        if recorded["state"] != state.value:
            problems_or_notes.append(
                f"the checkpoint says {recorded['state']}, the events say {state.value}"
            )
        for name in ("model_calls", "tool_calls", "tokens_total"):
            if int(recorded["usage"][name]) != getattr(usage, name):
                problems_or_notes.append(
                    f"usage.{name}: checkpoint {recorded['usage'][name]}, "
                    f"events {getattr(usage, name)}"
                )
        if Decimal(str(recorded["usage"]["dollars"])) != usage.dollars:
            problems_or_notes.append(
                f"usage.dollars: checkpoint {recorded['usage']['dollars']}, events {usage.dollars}"
            )
        plan: dict[str, Any] | None = snapshot.get("plan")
        for step in plan["steps"] if plan else []:
            replayed = steps.get(step["id"], "pending")
            if step["status"] != replayed:
                problems_or_notes.append(
                    f"step {step['index']}: checkpoint {step['status']}, events {replayed}"
                )
        if bool(snapshot.get("pending_approval")) != bool(requested):
            problems_or_notes.append("the pending approval in the checkpoint does not match")

    return ReplayReport(
        ok=not problems,
        event_count=len(events),
        state=state,
        usage=usage,
        step_statuses=steps,
        problems=problems,
        notes=notes,
    )
