import logging
import sys

import pytest

from aegis.log import RedactingFormatter, redact

# Obviously fake keys that have the SHAPE of real ones.
FAKE_OPENAI = "sk-proj-" + "A1b2C3d4" * 4
FAKE_ANTHROPIC = "sk-ant-api03-" + "Zy9X8w7V" * 4
FAKE_GROQ = "gsk_" + "x9Y8z7W6" * 5
FAKE_GOOGLE = "AIzaSy" + "a" * 33


@pytest.mark.parametrize("secret", [FAKE_OPENAI, FAKE_ANTHROPIC, FAKE_GROQ, FAKE_GOOGLE])
def test_redact_hides_provider_keys(secret: str) -> None:
    cleaned = redact(f"calling provider with key {secret} now")
    assert secret not in cleaned
    assert "[REDACTED]" in cleaned


def test_redact_hides_bearer_tokens_and_key_fields() -> None:
    text = 'Authorization: Bearer abc123.def456-ghi789 {"api_key": "plainsecretvalue"}'
    cleaned = redact(text)
    assert "abc123.def456-ghi789" not in cleaned
    assert "plainsecretvalue" not in cleaned


def test_redact_leaves_normal_text_alone() -> None:
    text = "task completed: 3 steps, 2 tool calls"
    assert redact(text) == text


def test_formatter_redacts_message_arguments_and_tracebacks() -> None:
    formatter = RedactingFormatter("%(message)s")

    record = logging.LogRecord("t", logging.INFO, __file__, 1, "key=%s", (FAKE_GROQ,), None)
    assert FAKE_GROQ not in formatter.format(record)

    try:
        raise RuntimeError(f"auth failed for {FAKE_ANTHROPIC}")
    except RuntimeError:
        error_record = logging.LogRecord(
            "t", logging.ERROR, __file__, 1, "boom", None, sys.exc_info()
        )
    assert FAKE_ANTHROPIC not in formatter.format(error_record)