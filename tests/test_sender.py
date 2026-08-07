"""TelegramSender tests: retry/backoff semantics, terminal failures."""

import time
from unittest.mock import AsyncMock

import pytest

from app.infrastructure.telegram.api import (
    RetryableTelegramError,
    TelegramApiError,
)
from app.infrastructure.telegram.rate_limit import RateLimiter
from app.infrastructure.telegram.sender import TelegramSender


def make_limiter() -> RateLimiter:
    return RateLimiter(global_per_sec=1_000_000, per_chat_per_sec=1_000_000, burst=1000)


def payload(**overrides) -> dict:
    base = {"type": "text", "text": "hi", "correlation_id": "c-1"}
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_success_first_attempt():
    api = AsyncMock(return_value={"message_id": 1})
    sender = TelegramSender(api, make_limiter())

    assert await sender.send(chat_id=111, payload=payload()) is True
    api.send_message.assert_awaited_once_with(chat_id=111, text="hi")


@pytest.mark.asyncio
async def test_retryable_error_then_success():
    api = AsyncMock()
    api.send_message.side_effect = [
        RetryableTelegramError("boom", retry_after=0.001),
        {"message_id": 2},
    ]
    sender = TelegramSender(api, make_limiter(), max_attempts=5)

    assert await sender.send(chat_id=111, payload=payload()) is True
    assert api.send_message.await_count == 2


@pytest.mark.asyncio
async def test_retry_after_is_honored():
    api = AsyncMock()
    api.send_message.side_effect = [
        RetryableTelegramError("slow down", retry_after=0.5),
        {"message_id": 3},
    ]
    sender = TelegramSender(api, make_limiter(), max_attempts=5)
    start = time.monotonic()
    await sender.send(chat_id=111, payload=payload())
    elapsed = time.monotonic() - start
    assert elapsed >= 0.5  # blocked at least the Telegram-specified cooldown


@pytest.mark.asyncio
async def test_attempts_exhausted_returns_false():
    api = AsyncMock()
    api.send_message.side_effect = RetryableTelegramError("boom")
    sender = TelegramSender(
        api, make_limiter(), max_attempts=3, base_delay_seconds=0.001, max_delay_seconds=0.01
    )

    assert await sender.send(chat_id=111, payload=payload()) is False
    assert api.send_message.await_count == 3


@pytest.mark.asyncio
async def test_terminal_error_no_retry():
    api = AsyncMock()
    api.send_message.side_effect = TelegramApiError("forbidden")
    sender = TelegramSender(api, make_limiter(), max_attempts=5)

    assert await sender.send(chat_id=111, payload=payload()) is False
    assert api.send_message.await_count == 1


@pytest.mark.asyncio
async def test_unsupported_payload_type_returns_false():
    api = AsyncMock()
    sender = TelegramSender(api, make_limiter())

    assert await sender.send(chat_id=111, payload={"type": "sticker"}) is False
    api.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_status_payload_is_sent_as_plain_text():
    api = AsyncMock()
    api.send_message.return_value = {"message_id": 9}
    sender = TelegramSender(api, make_limiter())

    result = await sender.send(
        chat_id=111, payload={"type": "status", "text": "⏳ thinking..."}, capture_message_id=True
    )

    assert result == 9
    api.send_message.assert_awaited_once_with(chat_id=111, text="⏳ thinking...")


@pytest.mark.asyncio
async def test_backoff_caps_at_max_delay():
    sender = TelegramSender(AsyncMock(), make_limiter(), max_delay_seconds=10)
    delay = sender._backoff(attempt=10, retry_after=None)
    assert 10 <= delay <= 11  # exponential capped at 10; jitter adds up to 1.0
    assert sender._backoff(attempt=1, retry_after=3) == 3


@pytest.mark.asyncio
async def test_capture_message_id_returns_id():
    api = AsyncMock()
    api.send_message.return_value = {"message_id": 42}
    sender = TelegramSender(api, make_limiter())

    result = await sender.send(chat_id=111, payload=payload(), capture_message_id=True)

    assert result == 42
    api.send_message.assert_awaited_once_with(chat_id=111, text="hi")


@pytest.mark.asyncio
async def test_capture_message_id_missing_returns_true():
    api = AsyncMock(return_value={})
    sender = TelegramSender(api, make_limiter())

    assert await sender.send(chat_id=111, payload=payload(), capture_message_id=True) is True


@pytest.mark.asyncio
async def test_capture_message_id_failure_returns_false():
    api = AsyncMock()
    api.send_message.side_effect = TelegramApiError("forbidden")
    sender = TelegramSender(api, make_limiter())

    assert await sender.send(chat_id=111, payload=payload(), capture_message_id=True) is False


@pytest.mark.asyncio
async def test_delete_message_success():
    api = AsyncMock()
    sender = TelegramSender(api, make_limiter())

    assert await sender.delete_message(chat_id=111, message_id=42) is True
    api.delete_message.assert_awaited_once_with(chat_id=111, message_id=42)


@pytest.mark.asyncio
async def test_delete_message_already_gone_is_success():
    api = AsyncMock()
    api.delete_message.side_effect = TelegramApiError("message to delete not found")
    sender = TelegramSender(api, make_limiter())

    assert await sender.delete_message(chat_id=111, message_id=42) is True


@pytest.mark.asyncio
async def test_delete_message_retries_then_gives_up():
    api = AsyncMock()
    api.delete_message.side_effect = RetryableTelegramError("flood", retry_after=0.001)
    sender = TelegramSender(
        api, make_limiter(), max_attempts=3, base_delay_seconds=0.001, max_delay_seconds=0.01
    )

    assert await sender.delete_message(chat_id=111, message_id=42) is False
    assert api.delete_message.await_count == 3
