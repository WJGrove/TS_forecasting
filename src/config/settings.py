from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Everything with APP_ prefix will map to these fields
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        extra="ignore",
    )

    environment: str = "dev"
    log_level: str = "INFO"
    db_url: str = "sqlite:///./data/dev.db"
    openai_api_key: str | None = None

@lru_cache
def get_settings() -> "Settings":
    return Settings()

# convenient singleton import
settings = get_settings()
