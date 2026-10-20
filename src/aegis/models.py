from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_wall_clock_s: int = Field(default=3600, gt=0)
    max_model_calls: int = Field(default=200, gt=0)
    max_tool_calls: int = Field(default=500, gt=0)
    max_tokens_total: int = Field(default=2_000_000, gt=0)
    max_dollars: Decimal = Field(default=Decimal("5.00"), gt=0)


class BudgetUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wall_clock_s: float = Field(default=0.0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    tokens_total: int = Field(default=0, ge=0)
    dollars: Decimal = Field(default=Decimal("0"), ge=0)


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    objective: str
    state: TaskState
    budget: Budget
    usage: BudgetUsage
    workspace_path: Path
    created_at: datetime
    updated_at: datetime
    plan_id: str | None = None
    last_error: str | None = None
    verification_result_id: str | None = None  # added: §5.1 invariant needs it


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    task_id: str
    step_id: str | None = None
    tool_name: str
    arguments: dict[str, Any]
    requested_at: datetime
    idempotency_key: str | None = None


class EventType(StrEnum):
    TASK_CREATED = "task_created"
    STATE_CHANGED = "state_changed"
    PLAN_CREATED = "plan_created"
    PLAN_REVISED = "plan_revised"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    EVIDENCE_ADDED = "evidence_added"
    VERIFICATION = "verification"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    BUDGET_EXCEEDED = "budget_exceeded"
    ERROR = "error"


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int  # autoincrement
    task_id: str
    type: EventType
    payload: dict[str, Any]
    created_at: datetime
    correlation_id: str | None = None  # ties related events (e.g. call -> result)


class ToolOutcome(StrEnum):
    """How one tool call ended. Anything other than OK is recorded as a failure."""

    OK = "ok"
    TOOL_NOT_FOUND = "tool_not_found"
    INVALID_ARGUMENTS = "invalid_arguments"
    POLICY_VIOLATION = "policy_violation"
    APPROVAL_REQUIRED = "approval_required"
    TOOL_ERROR = "tool_error"  # the tool ran and reported an expected failure
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"  # a bug in the tool, never the model's fault


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call: ToolCall
    started_at: datetime
    finished_at: datetime
    outcome: ToolOutcome  # added: lets the controller count policy violations (section 13.2)
    success: bool
    result_model: str | None = None  # name of the result Pydantic model
    result: dict[str, Any] | None = None
    error: str | None = None
    truncated: bool = False


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    summary: str  # short, for model context
    detail_ref: str | None = None  # pointer to the full record (the TOOL_RESULT event id)
    artifact_ids: list[str] = Field(default_factory=list)  