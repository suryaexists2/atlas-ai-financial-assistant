"""Application configuration.

All runtime settings live here, sourced from environment variables or `.env`.
No module reads `os.environ` directly — everything funnels through `Settings`.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Atlas"
    app_version: str = "0.1.0"
    app_env: Literal["dev", "test", "prod"] = "dev"
    debug: bool = False
    log_level: str = "INFO"

    # Local dev default; production overrides with Postgres (Supabase/Neon).
    database_url: str = "sqlite+aiosqlite:///./atlas_dev.db"

    # Database pool tuning (Postgres).
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_pre_ping: bool = True

    telegram_bot_token: str | None = Field(default=None, repr=False)
    telegram_webhook_secret: str | None = Field(default=None, repr=False)
    public_base_url: str | None = None

    llm_provider: Literal["openrouter", "openai", "anthropic"] = "openrouter"
    llm_model: str = "meta-llama/llama-3.3-70b-instruct"
    # Models tried in order when the primary model/route fails (per turn).
    llm_fallback_models: list[str] = Field(
        default_factory=lambda: [
            "openai/gpt-4o-mini",
            "google/gemini-2.0-flash",
            "meta-llama/llama-3.1-8b-instruct",
        ]
    )
    llm_max_tokens: int = 600
    llm_temperature: float = 0.3
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    openrouter_api_key: str | None = Field(default=None, repr=False)
    openai_api_key: str | None = Field(default=None, repr=False)
    anthropic_api_key: str | None = Field(default=None, repr=False)
    embeddings_provider: Literal["openai"] = "openai"

    sec_user_agent: str = "AtlasHackathon contact@example.com"
    finnhub_api_key: str | None = Field(default=None, repr=False)

    # --- Media / file ingestion ---
    file_max_bytes: int = 25_000_000
    ingestion_max_chars: int = 120_000  # budget cap for extracted text per turn
    ingestion_chunk_chars: int = 12_000  # chunk size for very large documents
    ingestion_excerpt_chars: int = 8_000  # excerpt shown to the model per media message
    stt_model: str = "openai/whisper-1"
    vision_model: str = "openai/gpt-4o-mini"
    stt_timeout_seconds: float = 90.0
    vision_timeout_seconds: float = 60.0
    media_download_timeout_seconds: float = 60.0

    # --- Agent Core ---
    agent_max_tool_rounds: int = 5
    agent_context_max_messages: int = 24
    agent_fallback_reply: str = "Sorry — I hit a temporary hiccup. Give me a moment and try again."
    # While true, fallback replies carry the real failure reason (debug only).
    agent_debug_reply_errors: bool = False

    # Scheduler: how long after a scheduled fire time a job may still run.
    scheduler_misfire_grace_seconds: int = 60
    # Scheduler: minimum interval between schedule-check cycles.
    scheduler_poll_interval_seconds: int = 15

    # --- Telegram delivery ---
    telegram_rate_limit_per_chat_per_sec: float = 1.0
    telegram_rate_limit_global_per_sec: float = 20.0
    telegram_rate_limit_burst: int = 5
    outbox_poll_interval_seconds: float = 0.5
    outbox_max_attempts: int = 5
    outbox_retry_base_seconds: float = 2.0
    outbox_retry_max_seconds: float = 300.0

    # Dev convenience: reply to every incoming message with a canned echo.
    echo_mode: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
