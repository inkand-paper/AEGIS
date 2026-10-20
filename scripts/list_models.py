"""List the model ids your API key can actually use (OpenAI-compatible providers only).

Run:  uv run python scripts/list_models.py
"""

import httpx

from aegis.config import get_settings
from aegis.log import redact
from aegis.providers import PROVIDERS


def main() -> int:
    settings = get_settings()
    preset = PROVIDERS.get(settings.provider.strip().lower())
    if preset is None or preset.kind != "openai":
        print("Model listing works for OpenAI-compatible providers only.")
        return 1
    base_url = (settings.base_url or preset.base_url).rstrip("/")
    key = settings.api_key.get_secret_value() if settings.api_key else ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    try:
        response = httpx.get(f"{base_url}/models", headers=headers, timeout=30)
    except httpx.HTTPError as exc:
        print(f"FAILED: could not reach {base_url}: {type(exc).__name__}")
        return 1
    if not response.is_success:
        print(f"FAILED: HTTP {response.status_code}: {redact(response.text[:300])}")
        return 1

    ids = sorted(item["id"] for item in response.json().get("data", []))
    print(f"{len(ids)} models available to this key:")
    for model_id in ids:
        print(" ", model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())