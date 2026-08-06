"""Registers the Telegram webhook (and prints its status).

Usage:
    python -m scripts.set_webhook

Requires TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_SECRET and PUBLIC_BASE_URL.
"""

import asyncio

from aiogram import Bot

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

logger = get_logger("set_webhook")


async def main() -> None:
    settings = get_settings()
    configure_logging("INFO")

    missing = [
        name
        for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_WEBHOOK_SECRET", "PUBLIC_BASE_URL")
        if getattr(settings, name.lower()) in (None, "")
    ]
    if missing:
        raise SystemExit(f"Missing required settings: {', '.join(missing)}")

    bot = Bot(token=settings.telegram_bot_token)
    url = f"{settings.public_base_url.rstrip('/')}/webhook/telegram"
    try:
        result = await bot.set_webhook(url=url, secret_token=settings.telegram_webhook_secret)
        info = await bot.get_webhook_info()
        logger.info(
            "webhook_registered",
            url=url,
            set_success=result,
            pending_update_count=info.pending_update_count,
        )
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
