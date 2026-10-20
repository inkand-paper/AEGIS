"""Initial planning (design doc section 10.1). Replanning arrives in a later phase.

The model is asked for a plan as JSON. Its answer is untrusted input: it is parsed and
validated (at least one step, a sane maximum, known dependencies, no cycles) before it is
allowed to become a Plan.
"""

import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from aegis.model_client import Message, ModelResponse
from aegis.models import Plan, Step, Task

logger = logging.getLogger(__name__)

MAX_STEPS = 10
MAX_STEP_CHARS = 500

# The controller supplies this so that every planning call is budgeted, recorded and cancellable.
Ask = Callable[[list[Message], list[dict[str, Any]], int], Awaitable[ModelResponse]]


class PlanError(Exception):
    """The model did not produce a usable plan."""


PLAN_SYSTEM_PROMPT = """You are the planning component of AEGIS, an autonomous agent.
Break the objective into a short ordered plan. Reply with ONLY a JSON object, no other text:

{"steps": [{"description": "...", "depends_on": [1]}]}

Rules:
- 1 to 10 steps. Each step is one concrete action that can be done with the tools below.
- Use the tools for facts and calculations. Never plan to guess something a tool can compute.
- The last step must produce the final answer to the objective.
- depends_on lists the 1-based numbers of steps that must finish first. Leave it out to run
  the step after the previous one.

Available tools:
{tools}"""


def _extract_json(text: str) -> Any:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise PlanError("the reply contains no JSON object")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise PlanError(f"the reply is not valid JSON: {exc.msg}") from None


def build_plan(task_id: str, data: Any) -> Plan:
    """Validate parsed JSON and turn it into a Plan. Raises PlanError with a fixable message."""
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        raise PlanError('expected a JSON object with a "steps" list')
    raw_steps: list[Any] = data["steps"]
    if not raw_steps:
        raise PlanError("the plan has no steps")
    if len(raw_steps) > MAX_STEPS:
        raise PlanError(f"the plan has {len(raw_steps)} steps; the maximum is {MAX_STEPS}")

    plan_id = uuid.uuid4().hex
    step_ids = [f"{plan_id[:8]}.s{number}" for number in range(1, len(raw_steps) + 1)]
    steps: list[Step] = []
    for number, raw in enumerate(raw_steps, start=1):
        if not isinstance(raw, dict):
            raise PlanError(f"step {number} must be an object")
        description = raw.get("description")
        if not isinstance(description, str) or not description.strip():
            raise PlanError(f"step {number} needs a non-empty description")
        if len(description) > MAX_STEP_CHARS:
            raise PlanError(f"step {number} description is longer than {MAX_STEP_CHARS} characters")
        if "depends_on" in raw:
            depends = raw["depends_on"]
            if not isinstance(depends, list) or not all(
                isinstance(d, int) and not isinstance(d, bool) for d in depends
            ):
                raise PlanError(f"step {number}: depends_on must be a list of step numbers")
            for dep in depends:
                if dep < 1 or dep > len(raw_steps):
                    raise PlanError(f"step {number} depends on unknown step {dep}")
                if dep == number:
                    raise PlanError(f"step {number} depends on itself")
            dep_numbers = sorted(set(depends))
        else:
            dep_numbers = [number - 1] if number > 1 else []
        steps.append(
            Step(
                id=step_ids[number - 1],
                plan_id=plan_id,
                index=number,
                description=description.strip(),
                depends_on=[step_ids[d - 1] for d in dep_numbers],
            )
        )

    plan = Plan(id=plan_id, task_id=task_id, steps=steps)
    execution_order(plan)  # raises PlanError on a cycle
    return plan


def execution_order(plan: Plan) -> list[Step]:
    """Steps in an order that respects depends_on, lowest index first. Raises on a cycle."""
    remaining = {step.id: step for step in plan.steps}
    done: set[str] = set()
    ordered: list[Step] = []
    while remaining:
        ready = [s for s in remaining.values() if all(d in done for d in s.depends_on)]
        if not ready:
            raise PlanError("the plan has a dependency cycle")
        step = min(ready, key=lambda s: s.index)
        ordered.append(step)
        done.add(step.id)
        del remaining[step.id]
    return ordered


class Planner:
    def __init__(self, *, max_tokens: int = 4000) -> None:
        self._max_tokens = max_tokens

    async def create_plan(
        self, task: Task, tool_listing: list[dict[str, Any]], ask: Ask
    ) -> Plan:
        tools = "\n".join(f"- {t['name']}: {t['description']}" for t in tool_listing)
        messages = [
            Message(role="system", content=PLAN_SYSTEM_PROMPT.replace("{tools}", tools)),
            Message(role="user", content=f"Objective: {task.objective}"),
        ]
        last_error = "unknown"
        for attempt in (1, 2):  # one retry, with the validation error fed back to the model
            response = await ask(messages, [], self._max_tokens)
            try:
                return build_plan(task.id, _extract_json(response.text or ""))
            except PlanError as exc:
                last_error = str(exc)
                logger.warning("plan attempt %d rejected: %s", attempt, last_error)
                messages.append(Message(role="assistant", content=response.text or ""))
                messages.append(
                    Message(
                        role="user",
                        content=f"That plan was rejected: {last_error}. "
                        "Reply again with ONLY the corrected JSON object.",
                    )
                )
        raise PlanError(last_error)
