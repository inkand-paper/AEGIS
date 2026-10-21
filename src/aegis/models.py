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
    acceptance: list[str] = Field(default_factory=list)  # added: criteria from --accept
    # added: side-effect classes the task declared up front ("workspace", "execute")
    allowed_effects: list[str] = Field(default_factory=list)


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
    MODEL_CALL = "model_call"  # added: usage must be recomputable from events (8.2)
    EVIDENCE_ADDED = "evidence_added"
    VERIFICATION = "verification"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    TASK_RESUMED = "task_resumed"  # added: a recovery after a crash, with what was found (8.2)
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
    INTERRUPTED = "interrupted"  # the process died before this call finished (recorded on resume)
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


class PendingApproval(BaseModel):
    """A tool call that is waiting for a human, plus what is needed to carry on afterwards.

    Phase 4 adds full crash recovery; until then this is the one piece of in-progress state that
    is saved, because without it `aegis approve` (a separate process) could not continue.
    """

    model_config = ConfigDict(extra="forbid")

    step_id: str
    tool_call_id: str  # the call that was blocked (its TOOL_RESULT says approval_required)
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    turn: int  # which model turn of the step this happened in
    # True when a human approved calls[0]. False for a recovered step: its calls still go
    # through the normal approval gate.
    approved: bool = True
    messages: list[dict[str, Any]]  # the step conversation so far
    calls: list[dict[str, Any]]  # the blocked call first, then any calls queued after it
    requested_at: datetime


class StepStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    plan_id: str
    index: int  # ordering within the plan
    description: str  # human-readable
    depends_on: list[str] = Field(default_factory=list)  # step ids
    status: StepStatus = StepStatus.PENDING
    evidence_ids: list[str] = Field(default_factory=list)  # verifier evidence records
    attempts: int = 0


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    task_id: str
    steps: list[Step]
    current_step_id: str | None = None
    revision: int = 1  # bumped on replan
    superseded_by: str | None = None


class EvidenceKind(StrEnum):
    TEST_PASS = "test_pass"
    FILE_EXISTS = "file_exists"
    FILE_DIFF = "file_diff"
    ARTIFACT_HASH = "artifact_hash"
    BROWSER_STATE = "browser_state"
    HUMAN_APPROVAL = "human_approval"
    COMMAND_EXIT_ZERO = "command_exit_zero"
    TOOL_RESULT = "tool_result"  # added: a tool really returned this (Phase 2 evidence)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    task_id: str
    step_id: str | None = None
    kind: EvidenceKind
    payload: dict[str, Any]  # kind-specific
    created_at: datetime
    source_tool_call_id: str


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    task_id: str
    passed: bool
    checks: list[dict[str, Any]]  # {name, passed, evidence_ids, message}
    created_at: datetime
    verifier_version: str
