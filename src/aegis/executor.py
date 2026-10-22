"""The dispatcher: the only path from a model's proposal to a side effect (principle P1).

Pipeline, in order (design doc section 6.2):
  record the proposed call -> lookup -> validate arguments -> policy -> approval gate
  -> execute with a timeout -> validate the result -> record the result

Every proposed call gets exactly one TOOL_CALL event and exactly one TOOL_RESULT event, even
when it is rejected. That is deliberate: rejected calls are how prompt injection attempts and
model mistakes show up in the audit trail.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ValidationError

from aegis.memory import EventLog
from aegis.models import EventType, Observation, ToolCall, ToolCallRecord, ToolOutcome
from aegis.tools.base import (
    PolicyError,
    ToolContext,
    ToolError,
    ToolNotFound,
    ToolRegistry,
    ToolUnavailable,
)

logger = logging.getLogger(__name__)


@dataclass
class _Result:
    outcome: ToolOutcome
    result: dict[str, Any] | None = None
    error: str | None = None
    result_model: str | None = None
    side_effects: str | None = None  # set on success: what this call was allowed to change
    idempotency_key: str | None = None  # set for successful non-idempotent calls
    deduplicated: bool = False  # True when the recorded result of an identical call was reused


@dataclass
class ExecutionResult:
    record: ToolCallRecord  # the full outcome, for the audit trail
    observation: Observation  # the short form that goes back into the model's context


def _describe_validation_error(exc: ValidationError) -> str:
    """Short and value-free, so the model can fix its call without echoing large inputs."""
    parts = []
    for item in exc.errors():
        where = ".".join(str(part) for part in item["loc"]) or "arguments"
        parts.append(f"{where}: {item['msg']}")
    return "; ".join(parts)


def idempotency_key(task_id: str, step_id: str | None, tool_name: str, arguments: Any) -> str:
    """Names one logical call: the same task, step, tool and (validated) arguments."""
    text = json.dumps([task_id, step_id, tool_name, arguments], sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def observation_text(
    outcome: ToolOutcome,
    result: dict[str, Any] | None,
    error: str | None,
    limit: int = 4000,
) -> tuple[str, bool]:
    """The text the model sees for one call. Long results are cut so one call cannot flood it."""
    if outcome is not ToolOutcome.OK:
        return f"ERROR ({outcome.value}): {error}", False
    text = json.dumps(result)
    if len(text) <= limit:
        return text, False
    return text[:limit] + " ...[truncated]", True


class Executor:
    def __init__(
        self,
        registry: ToolRegistry,
        events: EventLog,
        *,
        tool_timeout_s: float = 30.0,
        max_summary_chars: int = 4000,
    ) -> None:
        self._registry = registry
        self._events = events
        self._tool_timeout_s = tool_timeout_s
        self._max_summary_chars = max_summary_chars

    async def execute(
        self,
        call: ToolCall,
        ctx: ToolContext,
        *,
        arguments_error: str | None = None,
        approved: bool = False,
        refuse: str | None = None,
    ) -> ExecutionResult:
        """Run one proposed call. `arguments_error` is a parse failure from the model client.

        `approved=True` means a human granted this exact call, so the approval gate is skipped.
        Every other check (arguments, workspace policy) still runs.
        `refuse` is a reason the controller gives for not running this call at all (it already
        failed in this step and cannot succeed on a repeat). The call is still recorded (P5).
        """
        started_at = datetime.now(UTC)
        started = time.monotonic()
        self._events.append(
            ctx.task_id,
            EventType.TOOL_CALL,
            {
                "tool_call_id": call.id,
                "tool_name": call.tool_name,
                "arguments": call.arguments,
                "step_id": call.step_id,
                "approved": approved,
            },
            correlation_id=call.id,
        )
        try:
            result = await self._run(call, ctx, arguments_error, approved, refuse)
        except asyncio.CancelledError:
            # Keep the P5 invariant (every TOOL_CALL has a TOOL_RESULT) even when cancelled.
            self._record_result(
                call, ctx, _Result(ToolOutcome.CANCELLED, error="cancelled"), started
            )
            raise
        finished_at = datetime.now(UTC)
        result_event_id = self._record_result(call, ctx, result, started)

        summary, truncated = self._summarize(result)
        record = ToolCallRecord(
            call=call.model_copy(update={"idempotency_key": result.idempotency_key})
            if result.idempotency_key
            else call,
            started_at=started_at,
            finished_at=finished_at,
            outcome=result.outcome,
            success=result.outcome is ToolOutcome.OK,
            result_model=result.result_model,
            result=result.result,
            error=result.error,
            truncated=truncated,
        )
        observation = Observation(
            tool_call_id=call.id, summary=summary, detail_ref=str(result_event_id)
        )
        return ExecutionResult(record=record, observation=observation)

    # ------------------------------------------------------------------ pipeline

    async def _run(
        self,
        call: ToolCall,
        ctx: ToolContext,
        arguments_error: str | None,
        approved: bool,
        refuse: str | None,
    ) -> _Result:
        # 1. Lookup
        try:
            tool = self._registry.get(call.tool_name)
        except ToolNotFound:
            known = ", ".join(self._registry.names())
            return _Result(
                ToolOutcome.TOOL_NOT_FOUND,
                error=f"unknown tool {call.tool_name!r}. Available tools: {known}",
            )

        if refuse is not None:
            return _Result(ToolOutcome.REPEAT_BLOCKED, error=refuse)

        # 2. Argument validation
        if arguments_error is not None:
            return _Result(ToolOutcome.INVALID_ARGUMENTS, error=arguments_error)
        try:
            args = tool.arg_model.model_validate(call.arguments)
        except ValidationError as exc:
            return _Result(ToolOutcome.INVALID_ARGUMENTS, error=_describe_validation_error(exc))

        # 3. Policy
        if tool.requires_network and not ctx.allowed_hosts:
            return _Result(
                ToolOutcome.POLICY_VIOLATION,
                error="network tools are not allowed for this task (no host was allowed)",
            )
        for field in tool.path_fields:
            value = getattr(args, field)
            try:
                for path in value if isinstance(value, list) else [value]:
                    ctx.resolve(path)
            except PolicyError as exc:
                logger.warning("policy violation: tool=%s field=%s: %s", tool.name, field, exc)
                return _Result(ToolOutcome.POLICY_VIOLATION, error=str(exc))

        # 4. Approval gate. Fail closed: a side effect the task did not declare needs a human.
        effects = tool.effective_side_effects(args)
        if effects != "none" and effects not in ctx.allowed_effects and not approved:
            return _Result(
                ToolOutcome.APPROVAL_REQUIRED,
                error=f"{tool.name} has side effects ({effects}) and needs approval",
            )

        # 4b. Duplicate suppression. A call that is not idempotent is skipped when it repeats the
        # most recent change exactly and nothing else has changed since (for example a model
        # that re-proposes a write after a restart). Its recorded result is returned instead.
        key: str | None = None
        if not tool.idempotent and effects != "none":
            key = idempotency_key(
                ctx.task_id, call.step_id, tool.name, args.model_dump(mode="json")
            )
            previous = self._events.last_effectful_result(ctx.task_id)
            if previous is not None and previous.get("idempotency_key") == key:
                logger.info("tool %s: identical to the last change, not run again", tool.name)
                return _Result(
                    ToolOutcome.OK,
                    result=previous["result"],
                    result_model=previous["result_model"],
                    side_effects=effects,
                    idempotency_key=key,
                    deduplicated=True,
                )

        # 5. Execute (the TOOL_CALL event was already written, so a crash here is detectable)
        timeout_s = tool.timeout_s if tool.timeout_s is not None else self._tool_timeout_s
        try:
            raw = await asyncio.wait_for(tool.execute(args, ctx), timeout=timeout_s)
        except TimeoutError:
            return _Result(ToolOutcome.TIMEOUT, error=f"tool timed out after {timeout_s:g}s")
        except PolicyError as exc:
            logger.warning("policy violation inside tool=%s: %s", tool.name, exc)
            return _Result(ToolOutcome.POLICY_VIOLATION, error=str(exc))
        except ToolUnavailable as exc:
            return _Result(ToolOutcome.UNAVAILABLE, error=str(exc))
        except ToolError as exc:
            return _Result(ToolOutcome.TOOL_ERROR, error=str(exc))
        except Exception as exc:
            logger.exception("tool %s crashed", tool.name)
            return _Result(
                ToolOutcome.INTERNAL_ERROR,
                error=f"internal error in {tool.name}: {type(exc).__name__}",
            )

        # 6. Result validation. A wrong result type is a tool bug, not something to paper over.
        if not isinstance(raw, tool.result_model):
            logger.error("tool %s returned %s", tool.name, type(raw).__name__)
            return _Result(
                ToolOutcome.INTERNAL_ERROR,
                error=f"internal error in {tool.name}: result is not {tool.result_model.__name__}",
            )
        result: BaseModel = raw
        return _Result(
            ToolOutcome.OK,
            result=result.model_dump(mode="json"),
            result_model=tool.result_model.__name__,
            side_effects=effects,
            idempotency_key=key,
        )

    # ------------------------------------------------------------------ helpers

    def _record_result(
        self, call: ToolCall, ctx: ToolContext, result: _Result, started: float
    ) -> int:
        event = self._events.append(
            ctx.task_id,
            EventType.TOOL_RESULT,
            {
                "tool_call_id": call.id,
                "tool_name": call.tool_name,
                "outcome": result.outcome.value,
                "success": result.outcome is ToolOutcome.OK,
                "result_model": result.result_model,
                "result": result.result,
                "error": result.error,
                "side_effects": result.side_effects,
                "idempotency_key": result.idempotency_key,
                "deduplicated": result.deduplicated,
                "duration_ms": round((time.monotonic() - started) * 1000),
            },
            correlation_id=call.id,
        )
        return event.id

    def _summarize(self, result: _Result) -> tuple[str, bool]:
        return observation_text(
            result.outcome, result.result, result.error, self._max_summary_chars
        )
