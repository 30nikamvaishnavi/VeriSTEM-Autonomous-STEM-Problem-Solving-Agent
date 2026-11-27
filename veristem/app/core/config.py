from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "VeriSTEM"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Gemini API configuration (server-side only)
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-4-hydrogen", alias="GEMINI_MODEL")
    gemini_timeout_seconds: int = Field(default=60, alias="GEMINI_TIMEOUT_SECONDS")

    # Security & API authentication
    veristem_api_key: str = Field(default="", alias="VERISTEM_API_KEY")
    max_problem_chars: int = Field(default=8000, alias="MAX_PROBLEM_CHARS")
    rate_limit_requests_per_minute: int = Field(default=60, alias="RATE_LIMIT_REQUESTS_PER_MINUTE")

    # Persistence
    database_url: str = Field(
        default="sqlite+aiosqlite:///./veristem_data.db",
        alias="DATABASE_URL",
    )
    langgraph_checkpoint_db: str = Field(
        default="./veristem_checkpoints.db",
        alias="LANGGRAPH_CHECKPOINT_DB",
    )

    # Workflow & Sandbox
    max_correction_attempts: int = Field(default=3, ge=1, le=5, alias="MAX_CORRECTION_ATTEMPTS")
    workflow_timeout_seconds: int = Field(default=180, alias="WORKFLOW_TIMEOUT_SECONDS")
    sandbox_timeout_seconds: int = Field(default=10, ge=1, le=30, alias="SANDBOX_TIMEOUT_SECONDS")
    sandbox_memory_mb: int = Field(default=256, ge=64, le=1024, alias="SANDBOX_MEMORY_MB")

    # Telemetry
    otel_service_name: str = Field(default="veristem-api", alias="OTEL_SERVICE_NAME")


@lru_cache
def get_settings() -> Settings:
    return Settings()
