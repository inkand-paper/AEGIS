"""Provider-independent model client (design doc section 7).

Everything a provider returns is UNTRUSTED input. It is parsed into our own types and nothing
in this file can cause a side effect. The dispatcher (Phase 2) decides what actually happens.

Tools are passed to `generate()` in one neutral format, and each adapter translates it:

    {"name": "calculator", "description": "...", "parameters": {<JSON schema>}}
"""

import asyncio
import json
import logging
import random
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from aegis.log import redact
from aegis.models import BudgetUsage

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]
_ZERO = Decimal("0")


# --------------------------------------------------------------------------- errors


class ModelClientError(Exception):
    """Base class for everything this module raises."""


class ModelResponseError(ModelClientError):
    """The provider answered, but the answer is malformed or unusable."""


class ModelTransportError(ModelClientError):
    """The request failed after bounded retries, or failed in a way retrying cannot fix."""

    def __init__(self, message: str, *, status_code: int | None = None, attempts: int = 1) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.attempts = attempts


class ModelToolCallError(ModelTransportError):
    """The provider could not parse the tool call the model wrote (for example Groq's HTTP 400
    `tool_use_failed`: the arguments were not valid JSON, or 'Tool choice is none, but model
    called a tool' when it called a tool in a call that offered none). The model can be asked
    to try again, unlike a bad key or a missing model, so the controller treats this as a
    recoverable turn."""


# --------------------------------------------------------------------------- data contracts


class RawToolCall(BaseModel):
    """A tool call as the model proposed it. UNVALIDATED: the dispatcher checks it (P1)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    arguments_error: str | None = None  # set when the arguments were not a valid JSON object


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[RawToolCall] = Field(default_factory=list)  # on assistant messages
    tool_call_id: str | None = None  # on tool messages: which call this result answers


class ModelUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    dollars: Decimal = Field(default=_ZERO, ge=0)
    attempts: int = Field(default=1, ge=1)  # HTTP attempts; every retry counts as a model call

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    tool_calls: list[RawToolCall] = Field(default_factory=list)
    usage: ModelUsage
    finish_reason: str = "unknown"  # normalized: stop | length | tool_calls | ...


class ModelClient(Protocol):
    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> ModelResponse: ...


def record_usage(usage: BudgetUsage, response: ModelResponse) -> BudgetUsage:
    """Return a copy of `usage` with one model response added in (retries count as calls)."""
    used = response.usage
    return usage.model_copy(
        update={
            "model_calls": usage.model_calls + used.attempts,
            "tokens_total": usage.tokens_total + used.total_tokens,
            "dollars": usage.dollars + used.dollars,
        }
    )


# --------------------------------------------------------------------------- rate limiting


class RateLimiter:
    """Spaces calls at least `min_interval_s` apart so free tiers do not throttle us."""

    def __init__(
        self,
        min_interval_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = self._clock()
            start = max(now, self._next_allowed)
            if start > now:
                await self._sleep(start - now)
            self._next_allowed = start + self._min_interval_s


# --------------------------------------------------------------------------- shared plumbing


@dataclass
class _Parsed:
    text: str | None
    tool_calls: list[RawToolCall]
    input_tokens: int
    output_tokens: int
    finish_reason: str


def _is_retryable(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


_BODY_HINT = re.compile(r"try again in (\d+(?:\.\d+)?)\s*(ms|s)\b", re.IGNORECASE)


def _body_retry_hint(text: str) -> float | None:
    """The wait a provider names in an error body, e.g. 'Please try again in 18.09s'."""
    found = _BODY_HINT.search(text)
    if found is None:
        return None
    seconds = float(found.group(1))
    return seconds / 1000 if found.group(2).lower() == "ms" else seconds


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _raw_call(call_id: str | None, name: str, arguments: str | dict[str, Any]) -> RawToolCall:
    """Build a RawToolCall without ever crashing on bad model output."""
    cid = call_id or f"call_{uuid.uuid4().hex[:12]}"
    if isinstance(arguments, dict):
        return RawToolCall(id=cid, name=name, arguments=arguments)
    try:
        parsed = json.loads(arguments) if arguments.strip() else {}
    except json.JSONDecodeError as exc:
        error = f"arguments are not valid JSON: {exc.msg}"
        return RawToolCall(id=cid, name=name, arguments_error=error)
    if not isinstance(parsed, dict):
        return RawToolCall(id=cid, name=name, arguments_error="arguments JSON is not an object")
    return RawToolCall(id=cid, name=name, arguments=parsed)


class HttpModelClient(ABC):
    """Rate limit, timeout, bounded retry and redacted logging.

    Subclasses only translate between our neutral types and one provider's wire format.
    """

    provider_name = "http"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: SecretStr | None = None,
        input_price_per_mtok: Decimal = _ZERO,
        output_price_per_mtok: Decimal = _ZERO,
        rate_limiter: RateLimiter | None = None,
        max_attempts: int = 3,
        base_backoff_s: float = 0.5,
        max_backoff_s: float = 30.0,
        max_wait_s: float = 75.0,
        max_paced_waits: int = 8,
        max_paced_total_s: float = 180.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._input_price = input_price_per_mtok
        self._output_price = output_price_per_mtok
        self._rate_limiter = rate_limiter or RateLimiter(0.0, sleep=sleep)
        self._max_attempts = max_attempts
        self._base_backoff_s = base_backoff_s
        self._max_backoff_s = max_backoff_s
        # A per-minute limit never asks for more than about a minute; longer is a daily quota.
        self._max_wait_s = max_wait_s
        self._max_paced_waits = max_paced_waits
        self._max_paced_total_s = max_paced_total_s
        self._transport = transport
        self._sleep = sleep

    # -- what each provider adapter must supply --------------------------------------------

    @abstractmethod
    def _url(self) -> str: ...

    @abstractmethod
    def _headers(self) -> dict[str, str]: ...

    @abstractmethod
    def _build_payload(
        self, messages: list[Message], tools: list[dict[str, Any]], max_tokens: int
    ) -> dict[str, Any]: ...

    @abstractmethod
    def _parse(self, data: dict[str, Any]) -> _Parsed: ...

    # -- shared behaviour ------------------------------------------------------------------

    def _key(self) -> str:
        return self._api_key.get_secret_value() if self._api_key else ""

    def _cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        spent = Decimal(input_tokens) * self._input_price
        spent += Decimal(output_tokens) * self._output_price
        return spent / Decimal(1_000_000)

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        delay = min(self._max_backoff_s, self._base_backoff_s * 2.0 ** (attempt - 1))
        delay += random.uniform(0, delay * 0.25)  # jitter
        if retry_after is not None:
            delay = max(delay, retry_after)
        return delay

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> ModelResponse:
        payload = self._build_payload(messages, tools, max_tokens)
        logger.debug("model request: %s", json.dumps(payload, default=str))
        started = time.monotonic()
        data, attempts = await self._post_with_retry(payload, timeout_s)
        logger.debug("model response: %s", json.dumps(data, default=str))
        try:
            parsed = self._parse(data)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(part) for part in first["loc"])
            raise ModelResponseError(
                f"unexpected response shape at '{where}': {first['msg']}"
            ) from exc
        usage = ModelUsage(
            input_tokens=parsed.input_tokens,
            output_tokens=parsed.output_tokens,
            dollars=self._cost(parsed.input_tokens, parsed.output_tokens),
            attempts=attempts,
        )
        logger.info(
            "model call ok provider=%s model=%s attempts=%d in=%d out=%d finish=%s %.2fs",
            self.provider_name,
            self._model,
            attempts,
            usage.input_tokens,
            usage.output_tokens,
            parsed.finish_reason,
            time.monotonic() - started,
        )
        return ModelResponse(
            text=parsed.text,
            tool_calls=parsed.tool_calls,
            usage=usage,
            finish_reason=parsed.finish_reason,
        )

    async def _post_with_retry(
        self, payload: dict[str, Any], timeout_s: float
    ) -> tuple[dict[str, Any], int]:
        """POST with bounded retries.

        Two kinds of waiting are kept apart. A FAILURE (timeout, 5xx, a 429 with no stated wait)
        uses one of `max_attempts` and backs off. A PACED wait is a 429 in which the provider
        says exactly how long to wait (free tiers cap tokens per minute): the provider is not
        failing, it is asking us to slow down, so it does not use up an attempt. Paced waits are
        still bounded in count and in total time, so a call can never wait for ever.
        """
        last_error = "unknown error"
        last_status: int | None = None
        failures = 0
        tries = 0
        paced = 0
        paced_total = 0.0
        async with httpx.AsyncClient(transport=self._transport, timeout=timeout_s) as client:
            while True:
                await self._rate_limiter.wait()
                tries += 1
                retry_after: float | None = None
                try:
                    response = await client.post(self._url(), headers=self._headers(), json=payload)
                except httpx.TransportError as exc:  # timeouts and connection failures
                    last_status = None
                    last_error = f"{type(exc).__name__}: {exc}"
                else:
                    if response.is_success:
                        return self._decode_json(response), tries
                    last_status = response.status_code
                    last_error = f"HTTP {response.status_code}: {redact(response.text[:300])}"
                    if response.status_code == 400 and (
                        "tool_use_failed" in response.text or "Tool choice is none" in response.text
                    ):
                        raise ModelToolCallError(
                            last_error, status_code=last_status, attempts=tries
                        )
                    if not _is_retryable(response.status_code):
                        raise ModelTransportError(
                            last_error, status_code=last_status, attempts=tries
                        )
                    retry_after = _parse_retry_after(response.headers.get("retry-after"))
                    hinted = retry_after
                    if hinted is None and response.status_code == 429:
                        hinted = _body_retry_hint(response.text)
                    if hinted is not None and hinted > self._max_wait_s:
                        # e.g. a daily quota: waiting a minute will not help, so fail fast.
                        raise ModelTransportError(
                            f"rate limited; provider asks to wait {hinted:.0f}s. {last_error}",
                            status_code=last_status,
                            attempts=tries,
                        )
                    if (
                        response.status_code == 429
                        and hinted is not None
                        and paced < self._max_paced_waits
                        and paced_total + hinted <= self._max_paced_total_s
                    ):
                        paced += 1
                        paced_total += hinted
                        logger.warning(
                            "rate limited, pausing as the provider asked (%d/%d): %.1fs",
                            paced,
                            self._max_paced_waits,
                            hinted,
                        )
                        await self._sleep(hinted)
                        continue
                failures += 1
                if failures >= self._max_attempts:
                    break
                delay = self._backoff(failures, retry_after)
                logger.warning(
                    "model call failed (attempt %d/%d): %s; retrying in %.1fs",
                    failures,
                    self._max_attempts,
                    last_error,
                    delay,
                )
                await self._sleep(delay)
        raise ModelTransportError(
            f"gave up after {tries} attempts: {last_error}",
            status_code=last_status,
            attempts=tries,
        )

    @staticmethod
    def _decode_json(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise ModelResponseError(f"response body is not JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ModelResponseError("response JSON is not an object")
        return data


# --------------------------------------------------------------------------- OpenAI-compatible
# Works for OpenAI, Groq, Gemini (compat endpoint), OpenRouter, Ollama, vLLM, LM Studio, ...


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore")  # providers add fields; we only read what we need


class _OAIFunction(_Lenient):
    name: str
    arguments: str | dict[str, Any] = "{}"


class _OAIToolCall(_Lenient):
    id: str | None = None
    function: _OAIFunction


class _OAIMessage(_Lenient):
    content: str | None = None
    tool_calls: list[_OAIToolCall] | None = None


class _OAIChoice(_Lenient):
    message: _OAIMessage
    finish_reason: str | None = None


class _OAIUsage(_Lenient):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class _OAIResponse(_Lenient):
    choices: list[_OAIChoice] = Field(min_length=1)
    usage: _OAIUsage = Field(default_factory=_OAIUsage)


def _to_openai_message(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content or "",
        }
    out: dict[str, Any] = {"role": message.role}
    if message.role == "assistant" and message.tool_calls:
        out["content"] = message.content
        out["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    else:
        out["content"] = message.content or ""
    return out


class OpenAICompatClient(HttpModelClient):
    provider_name = "openai-compatible"

    def __init__(self, *, max_tokens_field: str = "max_tokens", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._max_tokens_field = max_tokens_field  # OpenAI itself wants "max_completion_tokens"

    def _url(self) -> str:
        return f"{self._base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._key():
            headers["Authorization"] = f"Bearer {self._key()}"
        return headers

    def _build_payload(
        self, messages: list[Message], tools: list[dict[str, Any]], max_tokens: int
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [_to_openai_message(m) for m in messages],
            self._max_tokens_field: max_tokens,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool["parameters"],
                    },
                }
                for tool in tools
            ]
        return payload

    def _parse(self, data: dict[str, Any]) -> _Parsed:
        response = _OAIResponse.model_validate(data)
        choice = response.choices[0]
        calls = [
            _raw_call(tc.id, tc.function.name, tc.function.arguments)
            for tc in choice.message.tool_calls or []
        ]
        return _Parsed(
            text=choice.message.content or None,
            tool_calls=calls,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            finish_reason=choice.finish_reason or "unknown",
        )


# --------------------------------------------------------------------------- Anthropic (native)

ANTHROPIC_VERSION = "2023-06-01"
_ANTHROPIC_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
}


class _AnthBlock(_Lenient):
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None


class _AnthUsage(_Lenient):
    input_tokens: int = 0
    output_tokens: int = 0


class _AnthResponse(_Lenient):
    content: list[_AnthBlock]
    stop_reason: str | None = None
    usage: _AnthUsage = Field(default_factory=_AnthUsage)


def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue  # system text goes in a separate top-level field
        if message.role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content or "",
            }
            last = out[-1] if out else None
            # Results answering one assistant turn must share ONE user message.
            if (
                last is not None
                and last["role"] == "user"
                and isinstance(last["content"], list)
                and last["content"]
                and last["content"][0].get("type") == "tool_result"
            ):
                last["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif message.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                for call in message.tool_calls
            )
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": "user", "content": message.content or ""})
    return out


class AnthropicClient(HttpModelClient):
    provider_name = "anthropic"

    def _url(self) -> str:
        return f"{self._base_url}/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "x-api-key": self._key(),
        }

    def _build_payload(
        self, messages: list[Message], tools: list[dict[str, Any]], max_tokens: int
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": _to_anthropic_messages(messages),
        }
        system = "\n\n".join(m.content for m in messages if m.role == "system" and m.content)
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool["parameters"],
                }
                for tool in tools
            ]
        return payload

    def _parse(self, data: dict[str, Any]) -> _Parsed:
        response = _AnthResponse.model_validate(data)
        text = "".join(b.text or "" for b in response.content if b.type == "text") or None
        calls: list[RawToolCall] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if not block.name:
                raise ModelResponseError("tool_use block has no name")
            calls.append(_raw_call(block.id, block.name, block.input or {}))
        reason = response.stop_reason or "unknown"
        return _Parsed(
            text=text,
            tool_calls=calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=_ANTHROPIC_FINISH.get(reason, reason),
        )
