import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict

from aegis.executor import Executor
from aegis.memory import EventLog
from aegis.models import EventType, ToolCall, ToolOutcome
from aegis.tools import default_registry
from aegis.tools.base import Tool, ToolContext, ToolError, ToolRegistry


class SpyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "ok"


class SpyResult(BaseModel):
    answer: str


class WrongResult(BaseModel):
    other: int = 1


class SpyTool(Tool[SpyArgs, SpyResult]):
    """Counts executions and can misbehave on request."""

    name = "spy"
    description = "test tool"
    arg_model = SpyArgs
    result_model = SpyResult
    side_effects: Literal["none", "workspace", "external"] = "none"
    requires_network = False
    idempotent = True

    def __init__(self, events: EventLog | None = None) -> None:
        self.runs = 0
        self.tool_call_events_seen_during_run: int | None = None
        self._events = events

    async def execute(self, args: SpyArgs, ctx: ToolContext) -> SpyResult:
        self.runs += 1
        if self._events is not None:
            seen = self._events.events(ctx.task_id, EventType.TOOL_CALL)
            self.tool_call_events_seen_during_run = len(seen)
        if args.mode == "tool_error":
            raise ToolError("expected failure")
        if args.mode == "crash":
            raise RuntimeError("secret internal detail")
        if args.mode == "slow":
            await asyncio.sleep(5)
        if args.mode == "wrong_type":
            return WrongResult()  # type: ignore[return-value]
        return SpyResult(answer="done")


class SideEffectTool(SpyTool):
    name = "writer"
    side_effects = "workspace"


class NetworkTool(SpyTool):
    name = "fetcher"
    requires_network = True


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    (path / "hello.txt").write_text("hello world", encoding="utf-8")
    return path


@pytest.fixture
def events() -> EventLog:
    return EventLog()


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(task_id="task1", workspace=workspace)


def make_call(tool_name: str, arguments: dict[str, Any], call_id: str = "c1") -> ToolCall:
    return ToolCall(
        id=call_id,
        task_id="task1",
        tool_name=tool_name,
        arguments=arguments,
        requested_at=datetime.now(UTC),
    )


def run(executor: Executor, call: ToolCall, ctx: ToolContext, **kwargs: Any):  # type: ignore[no-untyped-def]
    return asyncio.run(executor.execute(call, ctx, **kwargs))


def registry_with(*tools: Tool[Any, Any]) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


# ---------------------------------------------------------------- the happy path


def test_calculator_call_succeeds_and_is_recorded(events: EventLog, ctx: ToolContext) -> None:
    executor = Executor(default_registry(), events)
    result = run(executor, make_call("calculator", {"expression": "17 * 23"}), ctx)
    assert result.record.success is True
    assert result.record.outcome is ToolOutcome.OK
    assert result.record.result == {"expression": "17 * 23", "value": 391}
    assert result.record.result_model == "CalculatorResult"
    assert '"value": 391' in result.observation.summary
    kinds = [e.type for e in events.events("task1")]
    assert kinds == [EventType.TOOL_CALL, EventType.TOOL_RESULT]
    assert result.observation.detail_ref == str(events.events("task1")[1].id)


def test_fs_read_through_the_dispatcher(events: EventLog, ctx: ToolContext) -> None:
    executor = Executor(default_registry(), events)
    result = run(executor, make_call("fs_read", {"path": "hello.txt"}), ctx)
    assert result.record.success is True
    assert result.record.result is not None
    assert result.record.result["content"] == "hello world"


def test_the_call_is_recorded_before_the_tool_runs(events: EventLog, ctx: ToolContext) -> None:
    """Principle P4: the record exists before any side effect can happen."""
    spy = SpyTool(events)
    run(Executor(registry_with(spy), events), make_call("spy", {}), ctx)
    assert spy.tool_call_events_seen_during_run == 1


# ---------------------------------------------------------------- rejections: no side effect


def test_unknown_tool_is_rejected_and_lists_alternatives(
    events: EventLog, ctx: ToolContext
) -> None:
    result = run(Executor(default_registry(), events), make_call("rm_rf", {}), ctx)
    assert result.record.outcome is ToolOutcome.TOOL_NOT_FOUND
    assert result.record.success is False
    assert "calculator" in result.observation.summary


@pytest.mark.parametrize(
    "arguments",
    [{}, {"expression": 5}, {"expression": "1+1", "unexpected": True}, {"expression": ""}],
)
def test_bad_arguments_are_rejected_without_running(
    events: EventLog, ctx: ToolContext, arguments: dict[str, Any]
) -> None:
    result = run(Executor(default_registry(), events), make_call("calculator", arguments), ctx)
    assert result.record.outcome is ToolOutcome.INVALID_ARGUMENTS
    assert result.observation.summary.startswith("ERROR (invalid_arguments)")


def test_argument_parse_failure_from_the_model_client_is_reported(
    events: EventLog, ctx: ToolContext
) -> None:
    spy = SpyTool()
    result = run(
        Executor(registry_with(spy), events),
        make_call("spy", {}),
        ctx,
        arguments_error="arguments are not valid JSON: Unterminated string",
    )
    assert result.record.outcome is ToolOutcome.INVALID_ARGUMENTS
    assert "not valid JSON" in (result.record.error or "")
    assert spy.runs == 0


@pytest.mark.parametrize("path", ["../secret.txt", "../../etc/passwd", "/etc/passwd"])
def test_path_escape_is_a_recorded_policy_violation(
    events: EventLog, ctx: ToolContext, path: str
) -> None:
    result = run(Executor(default_registry(), events), make_call("fs_read", {"path": path}), ctx)
    assert result.record.outcome is ToolOutcome.POLICY_VIOLATION
    recorded = events.events("task1", EventType.TOOL_RESULT)[0]
    assert recorded.payload["outcome"] == "policy_violation"
    assert recorded.payload["success"] is False


def test_symlink_escape_through_the_dispatcher(
    events: EventLog, ctx: ToolContext, tmp_path: Path
) -> None:
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")
    try:
        (ctx.workspace / "link.txt").symlink_to(tmp_path / "secret.txt")
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow creating symlinks")
    call = make_call("fs_read", {"path": "link.txt"})
    result = run(Executor(default_registry(), events), call, ctx)
    assert result.record.outcome is ToolOutcome.POLICY_VIOLATION


def test_tools_with_side_effects_fail_closed(events: EventLog, ctx: ToolContext) -> None:
    writer = SideEffectTool()
    result = run(Executor(registry_with(writer), events), make_call("writer", {}), ctx)
    assert result.record.outcome is ToolOutcome.APPROVAL_REQUIRED
    assert writer.runs == 0


def test_network_tools_are_refused(events: EventLog, ctx: ToolContext) -> None:
    fetcher = NetworkTool()
    result = run(Executor(registry_with(fetcher), events), make_call("fetcher", {}), ctx)
    assert result.record.outcome is ToolOutcome.POLICY_VIOLATION
    assert fetcher.runs == 0


# ---------------------------------------------------------------- failures while running


def test_expected_tool_failure_is_reported_to_the_model(events: EventLog, ctx: ToolContext) -> None:
    result = run(
        Executor(default_registry(), events), make_call("fs_read", {"path": "nope.txt"}), ctx
    )
    assert result.record.outcome is ToolOutcome.TOOL_ERROR
    assert "not found" in result.observation.summary


def test_tool_crash_is_an_internal_error_and_hides_details(
    events: EventLog, ctx: ToolContext
) -> None:
    call = make_call("spy", {"mode": "crash"})
    result = run(Executor(registry_with(SpyTool()), events), call, ctx)
    assert result.record.outcome is ToolOutcome.INTERNAL_ERROR
    assert "secret internal detail" not in result.observation.summary
    assert "RuntimeError" in result.observation.summary


def test_wrong_result_type_is_an_internal_error(events: EventLog, ctx: ToolContext) -> None:
    result = run(
        Executor(registry_with(SpyTool()), events), make_call("spy", {"mode": "wrong_type"}), ctx
    )
    assert result.record.outcome is ToolOutcome.INTERNAL_ERROR
    assert result.record.result is None


def test_slow_tool_times_out(events: EventLog, ctx: ToolContext) -> None:
    executor = Executor(registry_with(SpyTool()), events, tool_timeout_s=0.05)
    result = run(executor, make_call("spy", {"mode": "slow"}), ctx)
    assert result.record.outcome is ToolOutcome.TIMEOUT


def test_cancellation_still_records_a_result(events: EventLog, ctx: ToolContext) -> None:
    executor = Executor(registry_with(SpyTool()), events, tool_timeout_s=30)

    async def scenario() -> None:
        task = asyncio.create_task(executor.execute(make_call("spy", {"mode": "slow"}), ctx))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert events.unmatched_tool_calls("task1") == []
    assert events.events("task1", EventType.TOOL_RESULT)[0].payload["outcome"] == "cancelled"


def test_long_results_are_truncated_for_the_model(events: EventLog, ctx: ToolContext) -> None:
    (ctx.workspace / "big.txt").write_text("x" * 10_000, encoding="utf-8")
    executor = Executor(default_registry(), events, max_summary_chars=500)
    result = run(executor, make_call("fs_read", {"path": "big.txt"}), ctx)
    assert result.record.success is True
    assert result.record.truncated is True
    assert result.observation.summary.endswith("...[truncated]")
    assert len(result.observation.summary) < 600
    full = events.events("task1", EventType.TOOL_RESULT)[0].payload["result"]
    assert len(full["content"]) == 10_000  # the audit trail keeps everything


# ---------------------------------------------------------------- the P5 invariant


def test_every_tool_call_has_exactly_one_tool_result(events: EventLog, ctx: ToolContext) -> None:
    registry = default_registry()
    registry.register(SpyTool())
    executor = Executor(registry, events, tool_timeout_s=0.05)
    attempts = [
        ("calculator", {"expression": "1+1"}),
        ("calculator", {}),
        ("nope", {}),
        ("fs_read", {"path": "../x"}),
        ("fs_read", {"path": "missing.txt"}),
        ("spy", {"mode": "crash"}),
        ("spy", {"mode": "slow"}),
    ]
    for number, (name, arguments) in enumerate(attempts):
        run(executor, make_call(name, arguments, call_id=f"c{number}"), ctx)

    calls = events.events("task1", EventType.TOOL_CALL)
    results = events.events("task1", EventType.TOOL_RESULT)
    assert len(calls) == len(results) == len(attempts)
    assert sorted(c.correlation_id or "" for c in calls) == sorted(
        r.correlation_id or "" for r in results
    )
    assert events.unmatched_tool_calls("task1") == []


# ---------------------------------------------------------------- registry


def test_registry_lists_tools_in_the_neutral_model_format() -> None:
    listing = default_registry().list_for_model()
    assert {tool["name"] for tool in listing} == {"calculator", "fs_list", "fs_read"}
    calculator = next(t for t in listing if t["name"] == "calculator")
    assert calculator["parameters"]["required"] == ["expression"]
    assert calculator["parameters"]["properties"]["expression"]["type"] == "string"


def test_registering_a_name_twice_fails() -> None:
    registry = ToolRegistry()
    registry.register(SpyTool())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(SpyTool())