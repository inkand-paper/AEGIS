from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from aegis.models import Budget


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AEGIS_",
        env_file=".env",
        extra="ignore",
    )

    provider: str = ""
    api_key: SecretStr | None = None  # SecretStr never prints its value (P10)
    data_dir: Path = Path("data")
    workspaces_dir: Path = Path("workspaces")
    default_budget: Budget = Field(default_factory=Budget)


@lru_cache
def get_settings() -> Settings:
    return Settings()