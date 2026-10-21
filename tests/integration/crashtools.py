"""Tools for the SIGKILL test. They write a visible trace of every real execution, so the test
can count how many times a side effect really happened."""

import asyncio
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict

from aegis.tools.base import Tool, ToolContext, ToolRegistry
from aegis.tools.calculator import CalculatorTool


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Done(BaseModel):
    note: str


class _Slow(Tool[NoArgs, Done]):
    description = "Records that it ran, then takes a long time when HANG is set."
    arg_model = NoArgs
    result_model = Done
    side_effects: Literal["none", "workspace", "execute", "external"] = "workspace"
    requires_network = False
    idempotent = False

    async def execute(self, args: NoArgs, ctx: ToolContext) -> Done:
        with (ctx.workspace / "trace.txt").open("a", encoding="utf-8") as handle:
            handle.write(f"{self.name} pid={os.getpid()}\n")
        if os.environ.get("HANG") == "1":
            await asyncio.sleep(120)  # the test kills the process while we sit here
        return Done(note=f"{self.name} finished")


class ChargeTool(_Slow):
    """Moves money: running it twice is a real bug. NOT idempotent."""

    name = "charge_card"
    idempotent = False


class LookupTool(_Slow):
    """Only reads: running it twice is harmless. Idempotent."""

    name = "lookup_price"
    side_effects = "none"
    idempotent = True


def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(ChargeTool())
    reg.register(LookupTool())
    return reg
