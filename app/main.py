"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, get_logger
from app.infrastructure.db.session import build_engine, build_session_factory, dispose_engine
from app.infrastructure.telegram.api import AiogramTelegramApi
from app.infrastructure.telegram.outbox_worker import OutboxWorker
from app.infrastructure.telegram.rate_limit import RateLimiter
from app.infrastructure.telegram.sender import TelegramSender
from app.interfaces.api.middleware import CorrelationMiddleware
from app.interfaces.api.routes import health
from app.interfaces.api.routes.webhook import router as webhook_router
from app.interfaces.telegram.processor import UpdateProcessor
from app.interfaces.telegram.responder import dev_echo_reply

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    engine = build_engine(settings)
    app.state.engine = engine
    app.state.session_factory = build_session_factory(engine)
    app.state.telegram_processor = None
    app.state.outbox_worker = None

    if settings.telegram_bot_token:
        from aiogram import Bot

        bot = Bot(token=settings.telegram_bot_token)
        app.state.telegram_bot = bot
        app.state.telegram_processor = UpdateProcessor(
            app.state.session_factory,
            dev_echo_reply,
            echo_mode=settings.echo_mode,
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
