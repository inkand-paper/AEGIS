import pytest

from aegis.config import Settings
from aegis.model_client import AnthropicClient, OpenAICompatClient
from aegis.providers import PROVIDERS, ProviderConfigError, build_client

FAKE_KEY = "gsk_" + "a1B2c3D4" * 4


def make_settings(**kwargs: object) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[arg-type]


def test_groq_uses_the_openai_compatible_client() -> None:
    client = build_client(make_settings(provider="groq", api_key=FAKE_KEY))
    assert isinstance(client, OpenAICompatClient)


def test_anthropic_uses_the_native_client() -> None:
    client = build_client(make_settings(provider="anthropic", api_key="sk-ant-" + "x" * 20))
    assert isinstance(client, AnthropicClient)


def test_provider_name_is_case_insensitive() -> None:
    assert build_client(make_settings(provider=" Groq ", api_key=FAKE_KEY))


def test_unknown_provider_lists_the_choices() -> None:
    with pytest.raises(ProviderConfigError, match="groq"):
        build_client(make_settings(provider="nonsense", api_key=FAKE_KEY))


def test_missing_key_gives_a_clear_error() -> None:
    with pytest.raises(ProviderConfigError, match="AEGIS_API_KEY"):
        build_client(make_settings(provider="openai"))


def test_custom_provider_needs_url_and_model() -> None:
    with pytest.raises(ProviderConfigError, match="AEGIS_BASE_URL"):
        build_client(make_settings(provider="custom"))
    client = build_client(
        make_settings(provider="custom", base_url="http://localhost:8000/v1", model="my-model")
    )
    assert isinstance(client, OpenAICompatClient)


def test_every_preset_can_be_built() -> None:
    for name in PROVIDERS:
        settings = make_settings(
            provider=name, api_key=FAKE_KEY, base_url="http://x.test/v1", model="m"
        )
        assert build_client(settings)