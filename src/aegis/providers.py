"""Provider presets and the factory that turns Settings into a ModelClient.

To support a new provider that speaks the OpenAI chat format, add one line to PROVIDERS
(or skip this file entirely and use AEGIS_PROVIDER=custom with a base URL and model).
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from aegis.config import Settings
from aegis.model_client import (
    AnthropicClient,
    ModelClient,
    OpenAICompatClient,
    RateLimiter,
    Sleep,
)


class ProviderConfigError(Exception):
    """Raised when the provider settings are incomplete or wrong. The message says how to fix it."""


@dataclass(frozen=True)
class ProviderPreset:
    kind: Literal["openai", "anthropic"]  # which wire format the adapter speaks
    base_url: str
    model: str  # default model id; override with AEGIS_MODEL (model ids change over time)
    min_interval_s: float = 0.0  # spacing between calls; protects free-tier rate limits
    max_tokens_field: str = "max_tokens"
    needs_key: bool = True


PROVIDERS: dict[str, ProviderPreset] = {
    # Free tier (about 30 requests per minute), so space calls 2 seconds apart.
    "groq": ProviderPreset(
        "openai", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b", min_interval_s=2.0
    ),
    # Free tier is limited to roughly 10 requests per minute. Set AEGIS_MIN_INTERVAL_S=0 if paid.
    "gemini": ProviderPreset(
        "openai",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-2.5-flash",
        min_interval_s=6.0,
    ),
    # Placeholders: untested until you add a key. Set AEGIS_MODEL to a model you have access to.
    "openai": ProviderPreset(
        "openai",
        "https://api.openai.com/v1",
        "gpt-4o-mini",
        max_tokens_field="max_completion_tokens",
    ),
    "anthropic": ProviderPreset("anthropic", "https://api.anthropic.com", "claude-sonnet-5-5"),
    # Needs AEGIS_MODEL (see openrouter.ai/models).
    "openrouter": ProviderPreset("openai", "https://openrouter.ai/api/v1", ""),
    # Local models: no key, no cost. Needs Ollama running.
    "ollama": ProviderPreset("openai", "http://localhost:11434/v1", "llama3.1", needs_key=False),
    # Any other OpenAI-compatible server. Needs AEGIS_BASE_URL and AEGIS_MODEL.
    "custom": ProviderPreset("openai", "", "", needs_key=False),
}


def build_client(
    settings: Settings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    sleep: Sleep = asyncio.sleep,
) -> ModelClient:
    name = settings.provider.strip().lower()
    preset = PROVIDERS.get(name)
    if preset is None:
        known = ", ".join(sorted(PROVIDERS))
        raise ProviderConfigError(f"Unknown AEGIS_PROVIDER {settings.provider!r}. Known: {known}")

    base_url = settings.base_url or preset.base_url
    model = settings.model or preset.model
    if not base_url or not model:
        raise ProviderConfigError(f"Provider {name!r} needs AEGIS_BASE_URL and AEGIS_MODEL in .env")

    key = settings.api_key
    if preset.needs_key and (key is None or not key.get_secret_value()):
        raise ProviderConfigError(f"Provider {name!r} needs AEGIS_API_KEY in your .env file")

    interval = (
        settings.min_interval_s if settings.min_interval_s is not None else preset.min_interval_s
    )
    common: dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "api_key": key,
        "input_price_per_mtok": settings.price_input_per_mtok,
        "output_price_per_mtok": settings.price_output_per_mtok,
        "rate_limiter": RateLimiter(interval, sleep=sleep),
        "transport": transport,
        "sleep": sleep,
    }
    if preset.kind == "anthropic":
        return AnthropicClient(**common)
    return OpenAICompatClient(max_tokens_field=preset.max_tokens_field, **common)