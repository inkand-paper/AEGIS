from decimal import Decimal

import pytest
from pydantic import ValidationError

from aegis.config import Settings
from aegis.models import Budget


def test_budget_rejects_negative_dollars() -> None:
    with pytest.raises(ValidationError):
        Budget(max_dollars=Decimal("-1"))


def test_budget_rejects_zero_tool_calls() -> None:
    with pytest.raises(ValidationError):
        Budget(max_tool_calls=0)


def test_budget_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Budget(max_banana=3)  # type: ignore[call-arg]


def test_settings_load_with_defaults() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.default_budget.max_dollars == Decimal("5.00")