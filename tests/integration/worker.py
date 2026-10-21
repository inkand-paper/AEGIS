"""A separate process that runs one task and gets killed by the test (design doc 12, Phase 4).

usage: python worker.py <data_dir> <tool_name>
It prints the task id, then hangs inside <tool_name>, waiting for SIGKILL.
"""

import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "unit"))

from crashtools import registry  # noqa: E402
from fakes import ScriptedModel, call_tools, plan_json, say  # noqa: E402

from aegis.controller import Controller  # noqa: E402
from aegis.memory import EventLog  # noqa: E402


def main() -> None:
    data_dir, tool_name = Path(sys.argv[1]), sys.argv[2]
    os.environ["HANG"] = "1"
    model = ScriptedModel(
        say(plan_json("Do the thing")),
        call_tools((tool_name, {})),
        say("never reached"),
    )
    controller = Controller(
        model,
        registry(),
        EventLog(data_dir / "aegis.db"),
        workspaces_dir=data_dir / "workspaces",
        lease_ttl_s=1.0,
    )
    task = controller.create_task("Do the thing", allowed_effects=["workspace"])
    print(task.id, flush=True)
    asyncio.run(controller.run(task.id))


if __name__ == "__main__":
    main()
