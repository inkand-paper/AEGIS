"""Live smoke test: one plain call and one tool-calling call against your configured provider.

Run:  uv run python scripts/smoke.py
"""

import asyncio

from aegis.config import get_settings
from aegis.log import setup_logging
from aegis.model_client import Message, ModelClientError, record_usage
from aegis.models import BudgetUsage
from aegis.providers import ProviderConfigError, build_client

CALCULATOR = {
    "name": "calculator",
    "description": "Evaluate a basic arithmetic expression and return the number.",
    "parameters": {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "e.g. 17 * 23"}},
        "required": ["expression"],
    },
}


async def main() -> int:
    setup_logging()
    settings = get_settings()
    usage = BudgetUsage()
    try:
        client = build_client(settings)

        plain = await client.generate(
            [Message(role="user", content="Reply with exactly one word: pong")],
            [],
            max_tokens=500,
            timeout_s=30,
        )
        usage = record_usage(usage, plain)
        print("plain call :", plain.text, "| finish:", plain.finish_reason)

        tool = await client.generate(
            [Message(role="user", content="What is 17 * 23? Use the calculator tool.")],
            [CALCULATOR],
            max_tokens=800,
            timeout_s=30,
        )
        usage = record_usage(usage, tool)
        print("tool call  :", [(c.name, c.arguments) for c in tool.tool_calls])
        print("tool finish:", tool.finish_reason)
    except (ProviderConfigError, ModelClientError) as exc:
        print("FAILED:", exc)
        return 1

    print("usage      :", usage.model_dump())
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))