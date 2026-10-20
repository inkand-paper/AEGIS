import asyncio
from pathlib import Path

import pytest

from aegis.tools.base import ToolContext, ToolError
from aegis.tools.calculator import CalculatorArgs, CalculatorTool

CTX = ToolContext(task_id="t", workspace=Path("."))


def calc(expression: str) -> int | float:
    args = CalculatorArgs(expression=expression)
    return asyncio.run(CalculatorTool().execute(args, CTX)).value


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("17 * 23", 391),
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("-2 ** 2", -4),
        ("2 ** 10", 1024),
        ("7 // 2", 3),
        ("7 % 4", 3),
        ("1 / 4", 0.25),
        ("  8 / 2  ", 4.0),
    ],
)
def test_arithmetic(expression: str, expected: int | float) -> None:
    assert calc(expression) == expected


def test_integer_results_stay_integers() -> None:
    assert isinstance(calc("17 * 23"), int)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('secret.txt')",
        "abs(-1)",
        "x + 1",
        "'a' * 3",
        "True + 1",
        "1j * 2",
        "[1, 2][0]",
        "(lambda: 1)()",
        "2 +",
        "1 / 0",
        "9 ** 9 ** 9",
        "10 ** 100000",
        "(-8) ** 0.5",
    ],
)
def test_anything_unsafe_or_invalid_is_rejected(expression: str) -> None:
    with pytest.raises(ToolError):
        calc(expression)


def test_argument_model_rejects_empty_and_oversized_input() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CalculatorArgs(expression="")
    with pytest.raises(ValidationError):
        CalculatorArgs(expression="1+" * 200 + "1")