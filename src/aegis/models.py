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