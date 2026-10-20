"""Watch the dispatcher handle good and bad tool calls. No model and no network needed.

Run:  uv run python scripts/demo_dispatch.py
"""

import asyncio
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aegis.executor import Executor
from aegis.memory import EventLog
from aegis.models import EventType, ToolCall
from aegis.tools import default_registry
from aegis.tools.base import ToolContext

PROPOSALS: list[tuple[str, dict[str, Any]]] = [
    ("calculator", {"expression": "17 * 23"}),
    ("fs_list", {"path": "."}),
    ("fs_read", {"path": "notes.txt"}),
    ("fs_read", {"path": "../../etc/passwd"}),  # path escape
    ("calculator", {"expression": "__import__('os').system('whoami')"}),  # code injection
    ("delete_everything", {}),  # no such tool
    ("calculator", {}),  # missing argument
]


async def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        workspace = Path(folder) / "workspace"
        workspace.mkdir()
        (workspace / "notes.txt").write_text("remember the milk", encoding="utf-8")

        events = EventLog()  # in memory for the demo
        executor = Executor(default_registry(), events)
        ctx = ToolContext(task_id="demo", workspace=workspace)

        for number, (tool_name, arguments) in enumerate(PROPOSALS, start=1):
            call = ToolCall(
                id=f"call-{number}",
                task_id="demo",
                tool_name=tool_name,
                arguments=arguments,
                requested_at=datetime.now(UTC),
            )
            result = await executor.execute(call, ctx)
            outcome = result.record.outcome.value
            print(f"{tool_name:<18} {outcome:<18} {result.observation.summary[:70]}")

        calls = events.events("demo", EventType.TOOL_CALL)
        results = events.events("demo", EventType.TOOL_RESULT)
        print(f"\nevents: {len(calls)} TOOL_CALL, {len(results)} TOOL_RESULT")
        print("calls with no result:", len(events.unmatched_tool_calls("demo")))


if __name__ == "__main__":
    asyncio.run(main())