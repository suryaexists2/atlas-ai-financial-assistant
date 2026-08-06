"""Dev polling runner: long-polls Telegram and feeds the same pipeline as the webhook.

Usage:
    python -m scripts.run_polling
"""

import asyncio

from aiogram import Bot, Dispatcher, types

from app.core.config import get_settings
from app.core.context import RequestContext, new_correlation_id, push_context
from app.core.logging import configure_logging, get_logger
from app.infrastructure.db.session import build_engine, build_session_factory, dispose_engine
from app.infrastructure.telegram.api import AiogramTelegramApi
from app.infrastructure.telegram.outbox_worker import OutboxWorker
from app.infrastructure.telegram.rate_limit import RateLimiter
from app.infrastructure.telegram.sender import TelegramSender
from app.interfaces.telegram.processor import UpdateProcessor
from app.interfaces.telegram.responder import dev_echo_reply

logger = get_logger("run_polling")


def build_worker(settings, session_factory, bot) -> OutboxWorker:
    rate_limiter = RateLimiter(
        global_per_sec=settings.telegram_rate_limit_global_per_sec,
        per_chat_per_sec=settings.telegram_rate_limit_per_chat_per_sec,
        burst=settings.telegram_rate_limit_burst,
    )
    sender = TelegramSender(
        AiogramTelegramApi(bot),
        rate_limiter,
        max_attempts=settings.outbox_max_attempts,
        base_delay_seconds=settings.outbox_retry_base_seconds,
        max_delay_seconds=settings.outbox_retry_max_seconds,
    )
    return OutboxWorker(
        session_factory,
        sender,
        poll_interval_seconds=settings.outbox_poll_interval_seconds,
        max_attempts=settings.outbox_max_attempts,
        retry_base_seconds=settings.outbox_retry_base_seconds,
        retry_max_seconds=settings.outbox_retry_max_seconds,
    )


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.telegram_bot_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is required for polling mode.")

    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    bot = Bot(token=settings.telegram_bot_token)
    worker = build_worker(settings, session_factory, bot)
    worker.start()

    processor = UpdateProcessor(
        session_factory,
        dev_echo_reply,
        echo_mode=settings.echo_mode,
    )

    dp = Dispatcher()

    @dp.update()
    async def handle_update(update: types.Update) -> None:
        correlation_id = new_correlation_id()
        push_context(RequestContext(correlation_id=correlation_id, source="polling"))
        try:
            await processor.process_update(
                update.model_dump(),
                source="polling",
                correlation_id=correlation_id,
            )
        except Exception:  # noqa: BLE001 - polling must never die on one update
            logger.exception("polling_update_failed", update_id=update.update_id)
        finally:
            from app.core.context import clear_context

            clear_context()

    logger.info("polling_started")
    try:
        await dp.start_polling(bot)
    finally:
        await worker.stop()
        await dispose_engine(engine)
        logger.info("polling_stopped")


if __name__ == "__main__":
    asyncio.run(main())
