import json
from typing import Any

import pytest

from aegis.planner import MAX_STEPS, PlanError, build_plan, execution_order


def steps(*items: dict[str, Any]) -> dict[str, Any]:
    return {"steps": list(items)}


def test_valid_plan_defaults_to_sequential_dependencies() -> None:
    plan = build_plan("t", steps({"description": "a"}, {"description": "b"}, {"description": "c"}))
    assert [s.index for s in plan.steps] == [1, 2, 3]
    assert plan.steps[0].depends_on == []
    assert plan.steps[1].depends_on == [plan.steps[0].id]
    assert plan.steps[2].depends_on == [plan.steps[1].id]
    assert all(s.plan_id == plan.id for s in plan.steps)
    assert plan.revision == 1


def test_explicit_dependencies_are_respected() -> None:
    plan = build_plan(
        "t",
        steps(
            {"description": "a"},
            {"description": "b", "depends_on": []},
            {"description": "c", "depends_on": [1, 2]},
        ),
    )
    assert plan.steps[1].depends_on == []
    assert plan.steps[2].depends_on == [plan.steps[0].id, plan.steps[1].id]


def test_execution_order_respects_dependencies() -> None:
    plan = build_plan(
        "t",
        steps(
            {"description": "needs 3", "depends_on": [3]},
            {"description": "independent", "depends_on": []},
            {"description": "needs 2", "depends_on": [2]},
        ),
    )
    assert [s.index for s in execution_order(plan)] == [2, 3, 1]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ("just text", "steps"),
        ({"steps": "nope"}, "steps"),
        ({"steps": []}, "no steps"),
        (steps({"description": ""}), "description"),
        (steps({"description": 5}), "description"),
        (steps({"nope": "x"}), "description"),
        (steps("not an object"), "object"),
        (steps({"description": "a", "depends_on": [9]}), "unknown step"),
        (steps({"description": "a", "depends_on": [0]}), "unknown step"),
        (steps({"description": "a", "depends_on": [1]}), "itself"),
        (steps({"description": "a", "depends_on": "1"}), "list of step numbers"),
        (steps({"description": "a", "depends_on": [True]}), "list of step numbers"),
        (steps({"description": "x" * 501}), "longer than"),
        (
            steps({"description": "a", "depends_on": [2]}, {"description": "b", "depends_on": [1]}),
            "cycle",
        ),
        ({"steps": [{"description": "s"}] * (MAX_STEPS + 1)}, "maximum"),
    ],
)
def test_invalid_plans_are_rejected_with_a_fixable_message(data: Any, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        build_plan("t", data)


def test_plan_survives_a_json_round_trip() -> None:
    plan = build_plan("t", steps({"description": "a"}, {"description": "b"}))
    restored = type(plan).model_validate(json.loads(plan.model_dump_json()))
    assert restored == plan


def test_the_planning_prompt_forbids_unrequested_side_effects() -> None:
    """A live model once added 'write the result to result.txt' to a pure calculation task."""
    from aegis.planner import PLAN_SYSTEM_PROMPT

    assert "Do NOT add steps that write, edit or run anything" in PLAN_SYSTEM_PROMPT


def test_the_planning_prompt_tells_the_model_not_to_pretend_about_missing_tools() -> None:
    """A live run planned 'compile with PyInstaller' although no tool can run it."""
    from aegis.planner import PLAN_SYSTEM_PROMPT

    assert "Only plan steps the tools below can really do" in PLAN_SYSTEM_PROMPT
    assert "do not pretend" in PLAN_SYSTEM_PROMPT


def test_the_agent_prompt_forbids_inventing_results() -> None:
    from aegis.controller import AGENT_SYSTEM_PROMPT

    assert "Never claim to have done what you did not do" in AGENT_SYSTEM_PROMPT
