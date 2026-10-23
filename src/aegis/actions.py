"""The action log: what the agent did, in order, readable by a person (Phase 7 exit criterion).

Built only from the event log, so it shows what really happened, including calls that were
refused, calls that waited for approval and calls that failed. Plain ASCII on purpose.
"""

import json
from datetime import UTC
from typing import Any

from aegis.models import Event, EventType

MAX_ARG_CHARS = 90


def _short(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=True) if not isinstance(value, str) else repr(value)
    return text if len(text) <= 40 else text[:37] + "..."


def describe_arguments(arguments: dict[str, Any]) -> str:
    text = " ".join(f"{key}={_short(value)}" for key, value in arguments.items())
    return text if len(text) <= MAX_ARG_CHARS else text[: MAX_ARG_CHARS - 3] + "..."


def render_actions(events: list[Event], *, only_prefix: str | None = None) -> str:
    """One line per tool call, plus a line for every approval and replan. Oldest first."""
    results: dict[str, Event] = {
        e.correlation_id: e for e in events if e.type is EventType.TOOL_RESULT and e.correlation_id
    }
    lines: list[str] = []
    for event in events:
        stamp = event.created_at.astimezone(UTC).strftime("%H:%M:%S")
        payload = event.payload
        if event.type is EventType.TOOL_CALL:
            name = str(payload.get("tool_name", "?"))
            if only_prefix and not name.startswith(only_prefix):
                continue
            outcome_event = results.get(event.correlation_id or "")
            if outcome_event is None:
                outcome = "no result (cut off)"
            else:
                outcome = str(outcome_event.payload.get("outcome", "?"))
                error = outcome_event.payload.get("error")
                if error:
                    outcome += f": {str(error)[:140]}"
            approved = " [approved by a human]" if payload.get("approved") else ""
            args = describe_arguments(payload.get("arguments") or {})
            lines.append(f"{stamp}  {name}({args}) -> {outcome}{approved}")
        elif event.type is EventType.APPROVAL_REQUESTED and not only_prefix:
            lines.append(f"{stamp}  -- waiting for a human: {payload.get('reason', '')}")
        elif event.type is EventType.PLAN_REVISED and not only_prefix:
            lines.append(f"{stamp}  -- plan revised ({payload.get('trigger')})")
    return "\n".join(lines) if lines else "(no actions recorded)"
