"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, get_logger
from app.infrastructure.db.session import build_engine, build_session_factory, dispose_engine
from app.infrastructure.llm.gateway import OpenRouterGateway
from app.infrastructure.providers.finnhub import FinnhubClient
from app.infrastructure.providers.sec import SecEdgarClient
from app.infrastructure.telegram.api import AiogramTelegramApi
from app.infrastructure.telegram.outbox_worker import OutboxWorker
from app.infrastructure.telegram.rate_limit import RateLimiter
from app.infrastructure.telegram.sender import TelegramSender
from app.interfaces.api.middleware import CorrelationMiddleware
from app.interfaces.api.routes import health
from app.interfaces.api.routes.webhook import router as webhook_router
from app.interfaces.telegram.processor import UpdateProcessor
from app.interfaces.telegram.responder import AgentComposer, EchoComposer

logger = get_logger(__name__)


def _build_sender(settings: Settings, bot) -> TelegramSender:
    rate_limiter = RateLimiter(
        global_per_sec=settings.telegram_rate_limit_global_per_sec,
        per_chat_per_sec=settings.telegram_rate_limit_per_chat_per_sec,
        burst=settings.telegram_rate_limit_burst,
    )
    return TelegramSender(
        AiogramTelegramApi(bot),
        rate_limiter,
        max_attempts=settings.outbox_max_attempts,
        base_delay_seconds=settings.outbox_retry_base_seconds,
        max_delay_seconds=settings.outbox_retry_max_seconds,
    )


def _build_worker(settings: Settings, session_factory, sender: TelegramSender) -> OutboxWorker:
    return OutboxWorker(
        session_factory,
        sender,
        poll_interval_seconds=settings.outbox_poll_interval_seconds,
        max_attempts=settings.outbox_max_attempts,
        retry_base_seconds=settings.outbox_retry_base_seconds,
        retry_max_seconds=settings.outbox_retry_max_seconds,
    )


def _build_media_ingestor(settings: Settings, bot):
    """Builds the media->text pipeline (download + parse + chunk). AI parsers
    (voice STT, image vision) activate only when credentials are present;
    pure-file parsers (txt/csv/json/pdf/docx/xlsx/md) always work.

    STT backend: `settings.stt_provider` selects Groq (free Whisper API, no
    OpenRouter balance gate) or OpenRouter (needs ~$0.50 account credit).
    Vision always uses OpenRouter when a key is set."""
    from app.application.ingestion.pipeline import IngestionPipeline
    from app.infrastructure.ingestion.downloader import TelegramFileFetcher
    from app.infrastructure.ingestion.media_ai import GroqSTT, OpenRouterMediaAI
    from app.infrastructure.ingestion.parsers import build_default_registry
    from app.interfaces.telegram.normalized import NormalizedMessage

    stt = None
    vision = None
    media_ai = None
    if settings.openrouter_api_key:
        media_ai = OpenRouterMediaAI(
            settings.openrouter_api_key,
            stt_model=settings.stt_model,
            vision_model=settings.vision_model,
            timeout_seconds=max(settings.stt_timeout_seconds, settings.vision_timeout_seconds),
        )
        vision = media_ai
    if settings.stt_provider == "groq":
        if settings.groq_api_key:
            stt = GroqSTT(
                settings.groq_api_key,
                model=settings.groq_stt_model,
                timeout_seconds=settings.stt_timeout_seconds,
            )
    elif media_ai is not None:
        stt = media_ai

    registry = build_default_registry(stt=stt, vision=vision)
    pipeline = IngestionPipeline(
        registry=registry,
        fetcher=TelegramFileFetcher(bot, max_bytes=settings.file_max_bytes),
        stt=stt,
        vision=vision,
        max_bytes=settings.file_max_bytes,
        max_chars=settings.ingestion_max_chars,
        chunk_chars=settings.ingestion_chunk_chars,
        excerpt_chars=settings.ingestion_excerpt_chars,
    )

    async def ingestor(message: NormalizedMessage):
        return await pipeline.process(
            file_id=message.media_file_id or "",
            mime_type=message.media_mime_type,
            filename=message.media_filename,
        )

    return ingestor


def _build_agent_composer(settings: Settings) -> AgentComposer | None:
    """Builds the agent composer when an LLM key is configured; else None."""
    from app.application.agent.core import AgentCore
    from app.application.agent.tools import default_registry

    if not settings.openrouter_api_key:
        logger.warning("agent_disabled_no_llm_key")
        return None

    gateway = OpenRouterGateway(
        settings.openrouter_api_key,
        settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        fallback_models=settings.llm_fallback_models,
    )
    finnhub = FinnhubClient(settings.finnhub_api_key) if settings.finnhub_api_key else None
    sec = SecEdgarClient(settings.sec_user_agent)
    from app.infrastructure.providers.google_sheets import GoogleSheetsClient
    from app.infrastructure.providers.stooq import MarketIndicesClient

    google_sheets: GoogleSheetsClient | None = GoogleSheetsClient()
    indices: MarketIndicesClient | None = MarketIndicesClient()
    agent = AgentCore(
        gateway,
        default_registry(),
        max_tool_rounds=settings.agent_max_tool_rounds,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        fallback_reply=settings.agent_fallback_reply,
        max_context_messages=settings.agent_context_max_messages,
        debug_reply_errors=settings.agent_debug_reply_errors,
    )
    return AgentComposer(
        agent, finnhub=finnhub, sec=sec, google_sheets=google_sheets, indices=indices
    )


def _build_scheduler(settings: Settings, session_factory):
    """Builds the proactive-intelligence scheduler worker, or None when the
    background feature set is unavailable. The scheduler only ever delivers via
    the durable Telegram outbox; it never talks to Telegram directly."""
    from app.application.intelligence import IntelligenceContext
    from app.application.intelligence.alerts import (
        run_filing_alerts,
        run_news_alerts,
        run_price_alerts,
    )
    from app.application.intelligence.briefing import daily_brief
    from app.application.intelligence.jobs import ensure_cycle_jobs
    from app.application.intelligence.reminders import fire_reminder
    from app.application.scheduling.worker import JobRunner, SchedulerWorker

    if not settings.database_url:
        return None

    finnhub = FinnhubClient(settings.finnhub_api_key) if settings.finnhub_api_key else None
    sec = SecEdgarClient(settings.sec_user_agent) if settings.sec_user_agent else None
    gateway = (
        OpenRouterGateway(
            settings.openrouter_api_key,
            settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            fallback_models=settings.llm_fallback_models,
        )
        if settings.openrouter_api_key
        else None
    )
    ctx = IntelligenceContext(finnhub=finnhub, sec=sec, gateway=gateway)
    runner = JobRunner(
        {
            "daily_brief": daily_brief,
            "reminder": fire_reminder,
            "price_alerts": run_price_alerts,
            "news_alerts": run_news_alerts,
            "filing_alerts": run_filing_alerts,
        }
    )
    scheduler = SchedulerWorker(
        session_factory,
        runner,
        context=ctx,
        poll_interval_seconds=settings.scheduler_poll_interval_seconds,
        misfire_grace_seconds=settings.scheduler_misfire_grace_seconds,
    )

    async def seed():
        from app.infrastructure.db.uow import UnitOfWork

        async with UnitOfWork(session_factory) as uow:
            await ensure_cycle_jobs(uow)

    return scheduler, seed


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    engine = build_engine(settings)
    app.state.engine = engine
    app.state.session_factory = build_session_factory(engine)
    app.state.telegram_processor = None
    app.state.outbox_worker = None
    app.state.scheduler_worker = None

    if settings.telegram_bot_token:
        from aiogram import Bot

        bot = Bot(token=settings.telegram_bot_token)
        app.state.telegram_bot = bot
        composer = (
            EchoComposer()
            if settings.echo_mode
            else (_build_agent_composer(settings) or EchoComposer())
        )
        app.state.telegram_processor = UpdateProcessor(
            app.state.session_factory,
            composer,
            echo_mode=True,  # echo_mode means "reply at all"; composer decides how
            fallback_reply=settings.agent_fallback_reply,
            media_ingestor=_build_media_ingestor(settings, bot),
        )
        worker = _build_worker(
            settings,
            app.state.session_factory,
            _build_sender(settings, bot),
        )
        app.state.outbox_worker = worker
        worker.start()
        logger.info("telegram_runtime_ready")
    else:
        logger.warning("telegram_not_configured_no_token")

    built = _build_scheduler(settings, app.state.session_factory)
    if built is not None:
        scheduler, seed = built
        try:
            await seed()
        except Exception:  # noqa: BLE001 - never block startup on job seeding
            logger.exception("scheduler_seed_failed")
        scheduler.start()
        app.state.scheduler_worker = scheduler
        logger.info("scheduler_runtime_ready")

    logger.info(
        "atlas_started",
        env=settings.app_env,
        database=settings.database_url.split("@")[-1].split("/")[-1],
        telegram_enabled=app.state.telegram_processor is not None,
    )
    try:
        yield
    finally:
        worker = getattr(app.state, "outbox_worker", None)
        if worker is not None:
            await worker.stop()
        scheduler = getattr(app.state, "scheduler_worker", None)
        if scheduler is not None:
            await scheduler.stop()
        await dispose_engine(engine)
        logger.info("atlas_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.app_env == "prod")

    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(CorrelationMiddleware)

    app.include_router(health.router)
    app.include_router(webhook_router)

    @app.exception_handler(AppError)
    async def app_error_handler(request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "message": exc.message, "details": exc.details},
        )

    return app


# Module-level instance for ASGI servers (uvicorn app.main:app).
app = create_app()
