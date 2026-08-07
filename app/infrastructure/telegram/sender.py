"""Outbound Telegram sender: rate-limited, retried with exponential backoff.

Single choke point for every message leaving the bot. Honors Telegram's
`retry_after` on 429 responses, backoffs on 5xx, and treats other 4xx as
terminal failures.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from app.core.logging import get_logger
from app.infrastructure.telegram.api import (
    RetryableTelegramError,
    TelegramApiError,
    TelegramApiPort,
)
from app.infrastructure.telegram.rate_limit import RateLimiter

logger = get_logger(__name__)


class TelegramSender:
    def __init__(
        self,
        api: TelegramApiPort,
        rate_limiter: RateLimiter,
        *,
        max_attempts: int = 5,
        base_delay_seconds: float = 2.0,
        max_delay_seconds: float = 300.0,
    ) -> None:
        self._api = api
        self._rate_limiter = rate_limiter
        self._max_attempts = max_attempts
        self._base_delay = base_delay_seconds
        self._max_delay = max_delay_seconds

    async def send(self, *, chat_id: int, payload: dict[str, Any]) -> bool:
        """Attempts to deliver a payload. Returns True on success."""
        message_type = payload.get("type", "text")
        text = payload.get("text")
        if message_type != "text" or text is None:
            logger.error("unsupported_outbound_payload", type=message_type)
            return False

        # Contextual, single-purpose inline button (e.g. "Connect Google" OAuth).
        # The outbox column is JSON, so nested dicts round-trip intact.
        reply_markup: dict[str, Any] | None = payload.get("reply_markup")
        if reply_markup is not None and not isinstance(reply_markup, dict):
            logger.error("unsupported_reply_markup")
            return False

        attempt = 0
        while attempt < self._max_attempts:
            attempt += 1
            await self._rate_limiter.acquire(chat_id)
            try:
                kwargs: dict[str, Any] = {"chat_id": chat_id, "text": text}
                if reply_markup is not None:
                    kwargs["reply_markup"] = reply_markup
                await self._api.send_message(**kwargs)
                logger.info(
                    "telegram_sent",
                    chat_id=chat_id,
                    attempt=attempt,
                    correlation_id=payload.get("correlation_id"),
                )
                return True
            except RetryableTelegramError as exc:
                if attempt >= self._max_attempts:
                    logger.warning("telegram_send_exhausted", chat_id=chat_id, attempt=attempt)
                    return False
                delay = self._backoff(attempt, exc.retry_after)
                logger.info(
                    "telegram_retry",
                    chat_id=chat_id,
                    attempt=attempt,
                    delay=round(delay, 2),
                    correlation_id=payload.get("correlation_id"),
                )
                await asyncio.sleep(delay)
            except TelegramApiError as exc:
                logger.warning("telegram_send_terminal", chat_id=chat_id, error=str(exc))
                return False
        return False

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return retry_after
        exponential = min(self._max_delay, self._base_delay * (2 ** (attempt - 1)))
        return exponential + random.uniform(0, min(1.0, exponential * 0.1))
