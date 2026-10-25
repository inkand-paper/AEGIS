import asyncio
import json
from decimal import Decimal
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from aegis.model_client import (
    AnthropicClient,
    Message,
    ModelResponseError,
    ModelToolCallError,
    ModelTransportError,
    OpenAICompatClient,
    RateLimiter,
    record_usage,
)
from aegis.models import BudgetUsage

CALC_TOOL = {
    "name": "calculator",
    "description": "Evaluate arithmetic.",
    "parameters": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
}
USER_HI = [Message(role="user", content="hi")]


def ok_body(
    content: str | None = "hello",
    tool_calls: list[dict[str, Any]] | None = None,
    finish: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
    }


def make_client(handler: Any, **kwargs: Any) -> tuple[OpenAICompatClient, list[float]]:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    client = OpenAICompatClient(
        base_url="https://example.test/v1",
        model="test-model",
        api_key=SecretStr("sk-test-secret-value"),
        transport=httpx.MockTransport(handler),
        sleep=fake_sleep,
        **kwargs,
    )
    return client, sleeps


def run(client: Any, messages: list[Message] = USER_HI, tools: list[dict[str, Any]] | None = None):
    return asyncio.run(client.generate(messages, tools or [], max_tokens=100, timeout_s=5))


# ---------------------------------------------------------------- Phase 1 exit criteria


def test_valid_response_becomes_a_validated_model_response() -> None:
    client, _ = make_client(lambda request: httpx.Response(200, json=ok_body()))
    response = run(client)
    assert response.text == "hello"
    assert response.finish_reason == "stop"
    assert response.usage.input_tokens == 1000
    assert response.usage.output_tokens == 500
    assert response.usage.attempts == 1


@pytest.mark.parametrize(
    "bad_response",
    [
        httpx.Response(200, json={"unexpected": True}),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, text="<html>not json</html>"),
        httpx.Response(200, json=["a", "list"]),
    ],
)
def test_malformed_response_raises_model_response_error(bad_response: httpx.Response) -> None:
    client, _ = make_client(lambda request: bad_response)
    with pytest.raises(ModelResponseError):
        run(client)


def test_429_is_retried_at_most_three_times_then_raises() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, json={"error": "slow down"})

    client, sleeps = make_client(handler)
    with pytest.raises(ModelTransportError) as info:
        run(client)
    assert len(calls) == 3
    assert info.value.status_code == 429
    assert info.value.attempts == 3
    assert len(sleeps) == 2  # a wait between attempts, none after the last


# ---------------------------------------------------------------- retry behaviour


def test_transient_failure_then_success_counts_every_attempt() -> None:
    responses = [httpx.Response(503, text="busy"), httpx.Response(200, json=ok_body())]
    client, _ = make_client(lambda request: responses.pop(0))
    response = run(client)
    assert response.usage.attempts == 2
    assert record_usage(BudgetUsage(), response).model_calls == 2


def test_timeouts_are_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ReadTimeout("too slow", request=request)

    client, _ = make_client(handler)
    with pytest.raises(ModelTransportError):
        run(client)
    assert len(calls) == 3


def test_auth_errors_fail_fast_without_retry() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": "bad key"})

    client, _ = make_client(handler)
    with pytest.raises(ModelTransportError) as info:
        run(client)
    assert len(calls) == 1
    assert info.value.status_code == 401


def test_a_tool_call_the_provider_could_not_parse_is_its_own_error_and_not_retried() -> None:
    """Live Groq run: HTTP 400 tool_use_failed when the model wrote broken JSON for fs_write."""
    calls: list[int] = []
    body = {
        "error": {
            "message": "Failed to parse tool call arguments as JSON",
            "type": "invalid_request_error",
            "code": "tool_use_failed",
            "failed_generation": '{"name": "fs_write", "arguments": {"path": "a.py", "content": "x',
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json=body)

    client, sleeps = make_client(handler)
    with pytest.raises(ModelToolCallError) as info:
        run(client)
    assert len(calls) == 1 and sleeps == []  # asking again with the same input would not help
    assert info.value.status_code == 400 and info.value.attempts == 1
    assert "sk-test-secret-value" not in str(info.value)


def test_other_bad_requests_are_still_plain_transport_errors() -> None:
    client, _ = make_client(lambda r: httpx.Response(400, json={"error": {"code": "other"}}))
    with pytest.raises(ModelTransportError) as info:
        run(client)
    assert not isinstance(info.value, ModelToolCallError)


def test_long_retry_after_fails_fast() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, headers={"Retry-After": "3600"}, json={"error": "daily quota"})

    client, _ = make_client(handler)
    with pytest.raises(ModelTransportError):
        run(client)
    assert len(calls) == 1


def test_short_retry_after_is_honoured() -> None:
    responses = [
        httpx.Response(429, headers={"Retry-After": "2"}, json={}),
        httpx.Response(200, json=ok_body()),
    ]
    client, sleeps = make_client(lambda request: responses.pop(0))
    run(client)
    assert sleeps == [2.0]


# ---------------------------------------------------------------- paced waits (free-tier limits)


def tpm_429(wait: str = "1.5s") -> httpx.Response:
    body = {
        "error": {
            "message": f"Rate limit reached on tokens per minute. Please try again in {wait}."
        }
    }
    return httpx.Response(429, json=body)


def test_a_rate_limit_that_names_its_wait_does_not_use_up_attempts() -> None:
    responses = [tpm_429() for _ in range(5)] + [httpx.Response(200, json=ok_body())]
    client, sleeps = make_client(lambda request: responses.pop(0))
    response = run(client)
    assert sleeps == [1.5] * 5  # waited exactly as asked, five times, and still succeeded
    assert response.usage.attempts == 6  # every HTTP call is still counted against the budget


def test_paced_waits_and_real_failures_are_counted_separately() -> None:
    responses = [
        tpm_429(),
        tpm_429(),
        httpx.Response(503, text="busy"),
        httpx.Response(503, text="busy"),
        httpx.Response(200, json=ok_body()),
    ]
    client, _ = make_client(lambda request: responses.pop(0))
    run(client)  # two failures (< 3 attempts) plus two paced waits: succeeds


def test_a_third_real_failure_still_ends_the_call_after_paced_waits() -> None:
    responses = [tpm_429()] + [httpx.Response(503, text="busy") for _ in range(3)]
    client, _ = make_client(lambda request: responses.pop(0))
    with pytest.raises(ModelTransportError) as info:
        run(client)
    assert info.value.attempts == 4 and not responses


def test_paced_waits_are_bounded_in_number() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return tpm_429("1s")

    client, sleeps = make_client(handler)
    with pytest.raises(ModelTransportError):
        run(client)
    assert len(sleeps) == 8 + 2  # 8 paced waits, then the normal backoff between 3 failures
    assert len(calls) == 8 + 3


def test_paced_waits_are_bounded_in_total_time() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return tpm_429("20s")

    client, sleeps = make_client(handler, max_paced_total_s=50)
    with pytest.raises(ModelTransportError):
        run(client)
    assert sleeps[:2] == [20.0, 20.0]  # a third 20 s would pass the 50 s allowance
    assert len(calls) == 2 + 3  # then the same 429s count as ordinary failures


def test_the_wait_is_read_in_milliseconds_too() -> None:
    responses = [tpm_429("682.5ms"), httpx.Response(200, json=ok_body())]
    client, sleeps = make_client(lambda request: responses.pop(0))
    run(client)
    assert sleeps == [0.6825]


def test_a_long_wait_named_in_the_body_fails_fast() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return tpm_429("120s")

    client, _ = make_client(handler)
    with pytest.raises(ModelTransportError, match="asks to wait 120s"):
        run(client)
    assert len(calls) == 1


def test_the_header_wins_over_the_body_and_a_5xx_is_never_paced() -> None:
    responses = [
        httpx.Response(
            429, headers={"Retry-After": "4"}, json={"error": {"message": "try again in 9s"}}
        ),
        httpx.Response(200, json=ok_body()),
    ]
    client, sleeps = make_client(lambda request: responses.pop(0))
    run(client)
    assert sleeps == [4.0]
    busy = [httpx.Response(503, headers={"Retry-After": "2"}, text="try again in 1s")] * 3
    client, _ = make_client(lambda request: busy[0])
    with pytest.raises(ModelTransportError) as info:
        run(client)
    assert info.value.attempts == 3  # a server error is a failure, whatever its text says


# ---------------------------------------------------------------- tool calls (untrusted)


def test_tool_call_arguments_are_parsed_but_not_trusted() -> None:
    body = ok_body(
        content=None,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "calculator", "arguments": '{"expression": "2+2"}'},
            }
        ],
        finish="tool_calls",
    )
    client, _ = make_client(lambda request: httpx.Response(200, json=body))
    response = run(client, tools=[CALC_TOOL])
    assert response.text is None
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "calculator"
    assert response.tool_calls[0].arguments == {"expression": "2+2"}


def test_bad_tool_arguments_do_not_crash_the_client() -> None:
    broken = {"name": "calculator", "arguments": '{"ex'}  # cut off mid-JSON
    body = ok_body(tool_calls=[{"id": "c1", "type": "function", "function": broken}])
    client, _ = make_client(lambda request: httpx.Response(200, json=body))
    call = run(client, tools=[CALC_TOOL]).tool_calls[0]
    assert call.arguments == {}
    assert call.arguments_error is not None  # the dispatcher will report this back to the model


def test_request_has_auth_header_and_openai_tool_format() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    client, _ = make_client(handler)
    run(client, tools=[CALC_TOOL])
    assert seen["auth"] == "Bearer sk-test-secret-value"
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["body"]["max_tokens"] == 100
    assert seen["body"]["tools"][0]["function"]["name"] == "calculator"


def test_max_tokens_field_can_be_renamed_for_openai() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    client, _ = make_client(handler, max_tokens_field="max_completion_tokens")
    run(client)
    assert seen["body"]["max_completion_tokens"] == 100
    assert "max_tokens" not in seen["body"]


# ---------------------------------------------------------------- cost and rate limiting


def test_cost_is_computed_and_recorded_into_budget_usage() -> None:
    client, _ = make_client(
        lambda request: httpx.Response(200, json=ok_body()),
        input_price_per_mtok=Decimal("1"),
        output_price_per_mtok=Decimal("2"),
    )
    response = run(client)
    assert response.usage.dollars == Decimal("0.002")  # (1000*1 + 500*2) / 1,000,000
    usage = record_usage(BudgetUsage(), response)
    assert (usage.model_calls, usage.tokens_total, usage.dollars) == (1, 1500, Decimal("0.002"))


def test_free_tier_costs_zero_dollars() -> None:
    client, _ = make_client(lambda request: httpx.Response(200, json=ok_body()))
    assert run(client).usage.dollars == Decimal("0")


def test_rate_limiter_spaces_calls() -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    limiter = RateLimiter(2.0, clock=lambda: 100.0, sleep=fake_sleep)

    async def two_calls() -> None:
        await limiter.wait()
        await limiter.wait()

    asyncio.run(two_calls())
    assert sleeps == [2.0]  # first call goes straight through, second waits


# ---------------------------------------------------------------- Anthropic adapter


def test_anthropic_request_translation_and_response_parsing() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "checking"},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "calculator",
                        "input": {"expression": "2+2"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    client = AnthropicClient(
        base_url="https://api.anthropic.test",
        model="claude-test",
        api_key=SecretStr("sk-ant-test-secret-value"),
        transport=httpx.MockTransport(handler),
    )
    history = [
        Message(role="system", content="be brief"),
        Message(role="user", content="add things"),
        Message(
            role="assistant",
            tool_calls=[
                {"id": "a", "name": "calculator", "arguments": {"expression": "1+1"}},  # type: ignore[list-item]
                {"id": "b", "name": "calculator", "arguments": {"expression": "2+2"}},  # type: ignore[list-item]
            ],
        ),
        Message(role="tool", tool_call_id="a", content="2"),
        Message(role="tool", tool_call_id="b", content="4"),
    ]
    response = run(client, messages=history, tools=[CALC_TOOL])

    body = seen["body"]
    assert seen["url"] == "https://api.anthropic.test/v1/messages"
    assert seen["headers"]["x-api-key"] == "sk-ant-test-secret-value"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert body["system"] == "be brief"
    assert body["tools"][0]["input_schema"] == CALC_TOOL["parameters"]
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert len(body["messages"][2]["content"]) == 2  # both tool results in ONE user message

    assert response.text == "checking"
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].arguments == {"expression": "2+2"}
    assert response.usage.total_tokens == 15
