"""Application configuration.

All runtime settings live here, sourced from environment variables or `.env`.
No module reads `os.environ` directly — everything funnels through `Settings`.
"""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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

    llm_provider: Literal["openrouter", "groq", "openai", "anthropic"] = "groq"
    llm_model: str = "meta-llama/llama-3.3-70b-instruct"
    # Models tried in order when the primary model/route fails (per turn).
    # Paid first for quality while credits exist; the `:free` routes cost $0 so
    # the bot keeps answering even when the account balance is exhausted.
    # Order matters: nvidia/nemotron-3-ultra-550b-a55b:free is a proven tool
    # caller, so it sits before the smaller/weaker free routes. super-120b is
    # excluded: it skips tool calls on market turns and fabricates quotes.
    llm_fallback_models: list[str] = Field(
        default_factory=lambda: [
            "openai/gpt-4o-mini",
            "google/gemma-4-31b-it:free",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "google/gemma-4-26b-a4b-it:free",
            "nvidia/nemotron-nano-9b-v2:free",
        ]
    )
    llm_max_tokens: int = 320
    llm_temperature: float = 0.3
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    # Groq chat stack (free tier): primary engine. `openai/gpt-oss-120b` is
    # Groq's current 120B open-weights route with reliable tool calling
    # (verified: emits get_market_quote correctly); the legacy llama-3.3-70b-
    # versatile route was hitting its per-day token cap. When the primary
    # rate-limits (it has its own 200K tokens/day bucket), the fallback is
    # `qwen/qwen3.6-27b`: Groq's multimodal chat route with a *separate* daily
    # bucket and full tool-calling/JSON support, so turns keep flowing even
    # when the gpt-oss bucket is exhausted. If both fail the gateway fails
    # over to the OpenRouter free chain instead.
    groq_llm_model: str = "openai/gpt-oss-120b"
    groq_llm_fallback: str | None = "qwen/qwen3.6-27b"
    # When true, the gateway periodically discovers new `:free` models from the
    # public OpenRouter catalogue and appends the compatible ones to the chain.
    llm_dynamic_free_models: bool = True
    # Compatibility floor for dynamically discovered models (see models_registry).
    llm_free_min_context: int = 32_000
    llm_free_min_completion: int = 600
    # How long a model that fails (402/400/404/429/empty) is skipped across turns.
    llm_model_skip_seconds: int = 600
    # Rate-limit (429) models recover in seconds, not minutes: a short penalty
    # keeps the primary engine in play under bursty traffic instead of banishing
    # it for the full skip window.
    llm_rate_limit_skip_seconds: int = 60
    # How often the free-model catalogue is re-fetched.
    llm_models_refresh_seconds: int = 21_600
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
    vision_model: str = "google/gemma-4-26b-a4b-it:free"
    # STT backend: "groq" (free Groq Whisper, needs a free GROQ_API_KEY from
    # console.groq.com) or "openrouter" (Whisper via OpenRouter, needs credit).
    stt_provider: Literal["openrouter", "groq"] = "groq"
    groq_api_key: str | None = Field(default=None, repr=False)
    # Drop-in swap as an existing single key now that the primary gets capped:
    # used by the key pool for chat + vision + speech on Groq. When present it
    # takes precedence; `GROQ_API_KEY` remains the single-key fallback.
    groq_api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list, repr=False)

    @field_validator("groq_api_keys", mode="before")
    @classmethod
    def _split_groq_api_keys(cls, value):
        # Env vars deliver plain comma-separated strings; pydantic-settings
        # would otherwise demand a JSON list and crash the boot.
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [item.strip() for item in value if item.strip()]
        return value

    @field_validator("groq_api_key", mode="after")
    @classmethod
    def _default_groq_api_key(cls, value):
        if not value:
            return "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"
        return value
    groq_stt_model: str = "whisper-large-v3-turbo"
    # Free-tier vision fallback used when the OpenRouter route fails (e.g. the
    # account balance is exhausted and every model 402s). qwen3.6-27b is the
    # current vision-capable model on GroqCloud (Llama 4 Scout/Maverick and the
    # llama-3.2 vision previews are no longer served).
    groq_vision_model: str = "qwen/qwen3.6-27b"
    stt_timeout_seconds: float = 90.0
    vision_timeout_seconds: float = 60.0
    media_download_timeout_seconds: float = 60.0

    # --- Google OAuth connectors (Gmail / Calendar / Drive) ---
    # Credentials from a Google Cloud "Web application" OAuth client. Redirect
    # URI is derived from public_base_url (/oauth/google/callback).
    google_oauth_client_id: str | None = Field(default=None, repr=False)
    google_oauth_client_secret: str | None = Field(default=None, repr=False)
    google_oauth_scopes: list[str] = Field(
        default_factory=lambda: [
            "openid",
            "email",
            "profile",
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/calendar.events",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
    )
    google_oauth_state_ttl_minutes: int = 10

    # --- Agent Core ---
    agent_max_tool_rounds: int = 1
    agent_context_max_messages: int = 12
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
    # Temporary "thinking" status bubbles: sent right after a user message,
    # deleted just before the final reply. TTL guards against a status that
    # never got a reply (e.g. the webhook died mid-turn).
    outbox_status_enabled: bool = True
    outbox_status_ttl_seconds: float = 600.0

    # Dev convenience: reply to every incoming message with a canned echo.
    echo_mode: bool = True

    # Onboarding: brand-new users are told the assistant is running on free
    # APIs in testing mode, so delays / rate limits are possible. Flip off
    # once the app leaves free-tier testing.
    onboarding_testing_notice: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
