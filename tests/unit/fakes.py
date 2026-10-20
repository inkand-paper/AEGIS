"""A scripted stand-in for the model, so controller tests need no network and no API key."""

import itertools
import json
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from aegis.model_client import Message, ModelResponse, ModelUsage, RawToolCall

_ids = itertools.count(1)

ScriptItem = ModelResponse | Exception | Callable[[list[Message]], ModelResponse]


def usage(tokens: int = 100, dollars: str = "0", attempts: int = 1) -> ModelUsage:
    half = tokens // 2
    return ModelUsage(
        input_tokens=half,
        output_tokens=tokens - half,
        dollars=Decimal(dollars),
        attempts=attempts,
    )


def say(text: str, **kwargs: Any) -> ModelResponse:
    return ModelResponse(text=text, usage=usage(**kwargs), finish_reason="stop")


def call_tools(*calls: tuple[str, dict[str, Any]], **kwargs: Any) -> ModelResponse:
    raw = [
        RawToolCall(id=f"p{next(_ids)}", name=name, arguments=arguments)
        for name, arguments in calls
    ]
    return ModelResponse(
        text=None, tool_calls=raw, usage=usage(**kwargs), finish_reason="tool_calls"
    )


def plan_json(*descriptions: str) -> str:
    return json.dumps({"steps": [{"description": text} for text in descriptions]})


class ScriptedModel:
    """Answers each generate() call with the next item of the script."""

    def __init__(self, *script: ScriptItem) -> None:
        self._script = list(script)
        self.calls: list[list[Message]] = []

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> ModelResponse:
        self.calls.append(list(messages))
        if not self._script:
            raise AssertionError("the model was called more often than the test scripted")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(messages)
        return item
