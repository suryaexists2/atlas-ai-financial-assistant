"""Telegram API port + aiogram adapter.

Outbound traffic is funneled through `TelegramApiPort` so tests can substitute
a fake and the sender layer never depends on aiogram internals. The adapter
translates aiogram exceptions into typed errors understood by the sender.
"""

from __future__ import annotations

from typing import Any, Protocol

from aiogram import Bot
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramRetryAfter,
    TelegramServerError,
    TelegramUnauthorizedError,
)
from httpx import TransportError


class TelegramApiError(Exception):
    """Terminal delivery failure (non-retryable or exhausted retries)."""


class RetryableTelegramError(TelegramApiError):
    """Transient failure (429 rate limit, 5xx) that may succeed on retry.

    When `retry_after` is set (from Telegram's 429 response), honor it.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TelegramApiPort(Protocol):
    async def send_message(
        self, *, chat_id: int, text: str, parse_mode: str | None = None
    ) -> dict[str, Any]: ...


class AiogramTelegramApi:
    """Adapter over the aiogram Bot."""

    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def send_message(
        self, *, chat_id: int, text: str, parse_mode: str | None = None
    ) -> dict[str, Any]:
        try:
            message = await self._bot.send_message(
                chat_id=chat_id, text=text, parse_mode=parse_mode
            )
        except TelegramRetryAfter as exc:
            raise RetryableTelegramError(str(exc), retry_after=float(exc.retry_after)) from exc
        except TelegramServerError as exc:
            raise RetryableTelegramError(str(exc)) from exc
        except TransportError as exc:
            raise RetryableTelegramError(str(exc)) from exc
        except (
            TelegramBadRequest,
            TelegramForbiddenError,
            TelegramUnauthorizedError,
        ) as exc:
            raise TelegramApiError(str(exc)) from exc
        return message.model_dump(exclude_none=True)
