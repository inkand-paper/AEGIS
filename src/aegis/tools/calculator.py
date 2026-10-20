"""A safe calculator. It parses the expression with `ast` and only evaluates arithmetic nodes.

It never calls eval(), so there is no way to reach names, attributes, imports or function calls.
"""

import ast
import math
import operator
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from aegis.tools.base import Tool, ToolContext, ToolError

MAX_INT_BITS = 10_000  # keeps results (and the time to compute them) bounded

_BINARY: dict[type[ast.operator], Callable[[int | float, int | float], int | float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY: dict[type[ast.unaryop], Callable[[int | float], int | float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str = Field(
        min_length=1,
        max_length=200,
        description="Arithmetic using numbers, + - * / // % ** and parentheses. Example: 17 * 23",
    )


class CalculatorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str
    value: int | float


def _check(value: int | float) -> int | float:
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise ToolError("the number is too large")
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError("the result is not a finite number")
    return value


def _power(base: int | float, exponent: int | float) -> int | float:
    if isinstance(base, int) and isinstance(exponent, int) and exponent >= 0:
        if base.bit_length() * exponent > MAX_INT_BITS:
            raise ToolError("the exponent is too large")
    result = base**exponent
    if isinstance(result, complex):
        raise ToolError("the result is not a real number")
    return result


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, int | float) and not isinstance(value, bool):
            return _check(value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _check(_UNARY[type(node.op)](_evaluate(node.operand)))
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            return _check(_power(left, right))
        if type(node.op) in _BINARY:
            return _check(_BINARY[type(node.op)](left, right))
    raise ToolError("only numbers, + - * / // % ** and parentheses are allowed")


class CalculatorTool(Tool[CalculatorArgs, CalculatorResult]):
    name = "calculator"
    description = "Evaluate an arithmetic expression exactly and return the number."
    arg_model = CalculatorArgs
    result_model = CalculatorResult
    side_effects = "none"
    requires_network = False
    idempotent = True

    async def execute(self, args: CalculatorArgs, ctx: ToolContext) -> CalculatorResult:
        try:
            tree = ast.parse(args.expression.strip(), mode="eval")
            value = _evaluate(tree)
        except SyntaxError:
            raise ToolError("that is not a valid arithmetic expression") from None
        except ZeroDivisionError:
            raise ToolError("division by zero") from None
        except (OverflowError, RecursionError):
            raise ToolError("the number is too large") from None
        return CalculatorResult(expression=args.expression, value=value)