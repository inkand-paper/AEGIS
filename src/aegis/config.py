from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from aegis.models import Budget


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AEGIS_",
        env_file=".env",
        env_ignore_empty=True,  # a blank "AEGIS_X=" line in .env counts as "not set"
        extra="ignore",
    )

    # --- model provider (Phase 1) ---
    provider: str = "groq"  # groq | gemini | openai | anthropic | openrouter | ollama | custom
    api_key: SecretStr | None = None  # SecretStr never prints its value (P10)
    base_url: str | None = None  # overrides the provider preset
    model: str | None = None  # overrides the provider preset
    min_interval_s: float | None = Field(default=None, ge=0)  # client-side spacing between calls
    price_input_per_mtok: Decimal = Field(default=Decimal("0"), ge=0)  # dollars per 1M tokens
    price_output_per_mtok: Decimal = Field(default=Decimal("0"), ge=0)

    # --- paths and budgets ---
    data_dir: Path = Path("data")
    workspaces_dir: Path = Path("workspaces")
    default_budget: Budget = Field(default_factory=Budget)

    # --- Docker sandbox for `submit --sandbox` (Phase 6) ---
    sandbox_image: str = "aegis-sandbox:latest"
    sandbox_memory_mb: int = Field(default=256, ge=16)
    sandbox_cpus: float = Field(default=1.0, gt=0)
    sandbox_pids: int = Field(default=64, ge=8)
    sandbox_timeout_s: float = Field(default=30.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
