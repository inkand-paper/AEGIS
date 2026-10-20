"""Logging with secret redaction (P10: secrets never appear in logs)."""

import logging
import re

REDACTED = "[REDACTED]"

_OPENAI_STYLE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")  # sk-..., sk-proj-..., sk-ant-api03-...
_GROQ = re.compile(r"gsk_[A-Za-z0-9]{8,}")
_GOOGLE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}")
_KEY_FIELD = re.compile(r"""(?i)("?(?:api[_-]?key|x-api-key)"?\s*[:=]\s*"?)[^\s",}]+""")

# (pattern, replacement). Patterns with a group keep the label and hide only the value.
_RULES: list[tuple[re.Pattern[str], str]] = [
    (_OPENAI_STYLE, REDACTED),
    (_GROQ, REDACTED),
    (_GOOGLE, REDACTED),
    (_BEARER, rf"\1{REDACTED}"),
    (_KEY_FIELD, rf"\1{REDACTED}"),
]


def redact(text: str) -> str:
    """Return `text` with anything that looks like an API key hidden."""
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


class RedactingFormatter(logging.Formatter):
    """Redacts the FINAL formatted line, so messages, arguments and tracebacks are all covered."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    # httpx logs every request URL at INFO. Our model client already logs one clean line per call.
    logging.getLogger("httpx").setLevel(logging.WARNING)