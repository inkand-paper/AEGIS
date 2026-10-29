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


def test_both_prompts_say_the_model_can_parse_and_count_data_itself() -> None:
    """Live run: the plan said 'cannot compute the sum, there is no tool to parse CSV'."""
    from aegis.planner import PLAN_SYSTEM_PROMPT, REVISE_SYSTEM_PROMPT

    for prompt in (PLAN_SYSTEM_PROMPT, REVISE_SYSTEM_PROMPT):
        assert 'a missing "CSV tool"' in prompt or 'missing "CSV tool"' in prompt
    assert "OUTSIDE capability" in PLAN_SYSTEM_PROMPT


def test_the_plan_prompt_lists_each_tools_arguments() -> None:
    from aegis.planner import format_tools
    from aegis.tools import default_registry

    text = format_tools(default_registry().list_for_model())
    line = next(x for x in text.splitlines() if x.startswith("- image_generate("))
    assert "width?" in line and "height?" in line  # optional, but the plan can now use them
    assert "prompt," in line and "path," in line  # required ones carry no question mark


def test_format_tools_handles_a_tool_with_no_arguments() -> None:
    from aegis.planner import format_tools

    out = format_tools([{"name": "ping", "description": "Ping.", "parameters": {}}])
    assert out == "- ping(): Ping."


def test_the_prompts_discourage_padding_and_wandering() -> None:
    """The first full benchmark run showed 7-step plans for a one-line fix, a step that only
    listed files, and a scratch file the agent made to explore something it was not asked."""
    from aegis.controller import AGENT_SYSTEM_PROMPT
    from aegis.planner import PLAN_SYSTEM_PROMPT

    assert "as few steps as the work needs" in PLAN_SYSTEM_PROMPT
    assert "Do not plan a step only to list files" in PLAN_SYSTEM_PROMPT
    assert "do not list the workspace" in AGENT_SYSTEM_PROMPT
    assert "do not create scratch files" in AGENT_SYSTEM_PROMPT


def test_test_writing_guidance_is_added_only_to_steps_that_write_tests() -> None:
    from aegis.controller import TEST_WRITING_PROMPT, _writes_tests

    assert _writes_tests("Create test_cart.py with pytest tests for total")
    assert _writes_tests("Write a new test file test_mathx.py")
    assert not _writes_tests("Run pytest on the project to verify that all tests pass")
    assert not _writes_tests("Read the contents of words.py")
    assert "value just outside" in TEST_WRITING_PROMPT and "equal" in TEST_WRITING_PROMPT


def _plan_text() -> str:
    return '{"steps": [{"description": "Read data.csv"}, {"description": "Write the sum"}]}'


def _task() -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(id="t1", objective="sum it")  # the planner reads only these two


def test_a_tool_call_during_planning_is_asked_again_with_a_reminder() -> None:
    import asyncio

    from aegis.model_client import ModelResponse, ModelToolCallError, ModelUsage
    from aegis.planner import NO_TOOL_IN_PLANNING, Planner

    seen: list[int] = []

    async def ask(messages: Any, tools: Any, max_tokens: int) -> Any:
        seen.append(len(messages))
        if len(seen) == 1:
            raise ModelToolCallError("HTTP 400: Tool choice is none, but model called a tool")
        assert messages[-1].content == NO_TOOL_IN_PLANNING
        return ModelResponse(text=_plan_text(), usage=ModelUsage())

    plan = asyncio.run(Planner().create_plan(_task(), [], ask))
    assert [s.description for s in plan.steps] == ["Read data.csv", "Write the sum"]
    assert seen == [2, 3]


def test_planning_gives_up_after_the_model_keeps_calling_tools() -> None:
    import asyncio

    from aegis.model_client import ModelToolCallError
    from aegis.planner import Planner

    calls: list[int] = []

    async def ask(messages: Any, tools: Any, max_tokens: int) -> Any:
        calls.append(1)
        raise ModelToolCallError("HTTP 400: Tool choice is none, but model called a tool")

    with pytest.raises(ModelToolCallError):
        asyncio.run(Planner().create_plan(_task(), [], ask))
    assert len(calls) == 3
