"""Planning (design doc section 10.1) and replanning (section 10.2).

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
from aegis.models import Plan, Step, StepStatus, Task

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
- Plan only what the objective asks for. Do NOT add steps that write, edit or run anything
  (files, tests, formatters) unless the objective says to. Reporting a result is done in the
  final answer, not by saving it to a file.
- You can read, parse, count, sort, filter and reformat text and data yourself, and do small
  arithmetic. A missing "CSV tool" or "sorting tool" is NOT a reason to give up: read the file
  with a tool, then do the work and write the result with a tool.
- Only plan steps the tools below can really do. A missing OUTSIDE capability is the limit: for
  example building an executable, installing software, running a program no tool can run, or
  using a website no tool can open. For that part, do not pretend: make the last step report
  plainly what could not be done and why.
- Use as few steps as the work needs: a small task is 2 or 3 steps. Reading a file belongs in
  the step that uses it, not in a step of its own. Do not plan a step only to list files or to
  confirm that something exists, and do not end with a step that re-reads or re-checks the work:
  AEGIS verifies the result itself.
- The last step must produce the final answer to the objective.
- depends_on lists the 1-based numbers of steps that must finish first. Leave it out to run
  the step after the previous one.

Available tools:
{tools}"""


REVISE_SYSTEM_PROMPT = """You are the planning component of AEGIS, an autonomous agent.
A plan was being carried out and it hit a problem. Write the REMAINING steps only. Steps that are
already done stay done and are not repeated. Reply with ONLY a JSON object, no other text:

{"steps": [{"description": "...", "depends_on": [1]}]}

Rules:
- 1 to 10 steps. Each step is one concrete action that can be done with the tools below.
- Do NOT repeat the approach that failed. Choose a different way, or a smaller piece of work.
- Only plan what the tools below can really do. You can parse, count, sort and reformat data
  yourself, so a missing "CSV tool" is not a limit. Only if the objective needs an outside
  capability no tool offers, make the single remaining step report plainly what could not be done
  and why. Do not pretend.
- The last step must produce the final answer to the objective.
- depends_on lists the 1-based numbers of the steps in THIS list that must finish first. Leave
  it out to run the step after the previous one.

Available tools:
{tools}"""

MAX_RESULT_CONTEXT_CHARS = 300


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


def format_tools(listing: list[dict[str, Any]]) -> str:
    """One line per tool with its argument names, so the plan can use what a tool really offers.
    (A live run planned 'image_generate cannot set the size' although it has width and height.)
    Optional arguments end with '?'."""
    lines = []
    for tool in listing:
        schema = tool.get("parameters") or {}
        required = set(schema.get("required", []))
        names = [
            name if name in required else f"{name}?" for name in (schema.get("properties") or {})
        ]
        lines.append(f"- {tool['name']}({', '.join(names)}): {tool['description']}")
    return "\n".join(lines)


class Planner:
    def __init__(self, *, max_tokens: int = 4000) -> None:
        self._max_tokens = max_tokens

    async def create_plan(self, task: Task, tool_listing: list[dict[str, Any]], ask: Ask) -> Plan:
        tools = format_tools(tool_listing)
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

    async def revise_plan(
        self,
        task: Task,
        current: Plan,
        step_results: dict[str, str],
        trigger: str,
        detail: str,
        tool_listing: list[dict[str, Any]],
        ask: Ask,
        *,
        revision: int,
    ) -> Plan:
        """Build the next plan: finished steps are kept as they are, the rest is re-planned."""
        done = [s for s in execution_order(current) if s.status is StepStatus.DONE]
        tools = format_tools(tool_listing)
        lines = [f"Objective: {task.objective}", "", "Steps already done:"]
        if done:
            for step in done:
                result = step_results.get(step.id, "")[:MAX_RESULT_CONTEXT_CHARS]
                lines.append(f"- {step.description} -> {result}")
        else:
            lines.append("- (none)")
        failed = [s for s in current.steps if s.status is not StepStatus.DONE]
        lines += ["", "Steps that were not finished:"]
        lines += [f"- {s.description}" for s in failed] or ["- (none)"]
        lines += ["", f"What went wrong ({trigger}): {detail[:600]}"]
        messages = [
            Message(role="system", content=REVISE_SYSTEM_PROMPT.replace("{tools}", tools)),
            Message(role="user", content="\n".join(lines)),
        ]
        last_error = "unknown"
        for attempt in (1, 2):
            response = await ask(messages, [], self._max_tokens)
            try:
                fresh = build_plan(task.id, _extract_json(response.text or ""))
                return merge_revision(current, done, fresh, revision)
            except PlanError as exc:
                last_error = str(exc)
                logger.warning("revised plan attempt %d rejected: %s", attempt, last_error)
                messages.append(Message(role="assistant", content=response.text or ""))
                messages.append(
                    Message(
                        role="user",
                        content=f"That plan was rejected: {last_error}. "
                        "Reply again with ONLY the corrected JSON object.",
                    )
                )
        raise PlanError(last_error)


def merge_revision(current: Plan, done: list[Step], fresh: Plan, revision: int) -> Plan:
    """Put the kept (done) steps in front of the new ones, under the new plan's id.

    New steps that depended on nothing now run after the last kept step, so the order of work
    stays one line from the start of the task to its final answer.
    """
    kept = [step.model_copy(update={"plan_id": fresh.id}) for step in done]
    anchor = [kept[-1].id] if kept else []
    offset = len(kept)
    new_steps = [
        step.model_copy(
            update={"index": step.index + offset, "depends_on": step.depends_on or anchor}
        )
        for step in fresh.steps
    ]
    for index, step in enumerate(kept, start=1):
        kept[index - 1] = step.model_copy(update={"index": index})
    plan = Plan(
        id=fresh.id,
        task_id=current.task_id,
        steps=[*kept, *new_steps],
        revision=revision,
    )
    execution_order(plan)
    return plan
