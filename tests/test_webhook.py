"""Webhook endpoint tests: secret validation + processor wiring."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings, get_settings
from app.main import create_app
from tests.conftest import tg_text_update


async def make_client(secret: str | None, processor: AsyncMock | None):
    settings = Settings(
        app_env="test",
        database_url="sqlite+aiosqlite://",
        telegram_webhook_secret=secret,
    )
    app = create_app(settings)
    if processor is not None:
        app.state.telegram_processor = processor
    app.dependency_overrides[get_settings] = lambda: settings
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    await client.__aenter__()
    return client


def update_payload() -> dict:
    return tg_text_update(update_id=1, chat_id=777, message_id=21, text="hello")


@pytest.mark.asyncio
async def test_missing_secret_returns_403():
    client = await make_client(secret=None, processor=AsyncMock(return_value=True))
    try:
        resp = await client.post("/webhook/telegram", json=update_payload())
        assert resp.status_code == 403
    finally:
        await client.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_wrong_secret_returns_403():
    client = await make_client(secret="correct", processor=AsyncMock(return_value=True))
    try:
        resp = await client.post(
            "/webhook/telegram",
            json=update_payload(),
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
        )
        assert resp.status_code == 403
    finally:
        await client.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_correct_secret_dispatches_to_processor():
    processor = AsyncMock(return_value=True)
    client = await make_client(secret="correct", processor=processor)
    try:
        resp = await client.post(
            "/webhook/telegram",
            json=update_payload(),
            headers={"X-Telegram-Bot-Api-Secret-Token": "correct"},
        )
        assert resp.status_code == 200
        processor.process_update.assert_awaited_once()
        _, kwargs = processor.process_update.await_args
        assert kwargs["source"] == "webhook"
        assert kwargs["correlation_id"]
    finally:
        await client.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_no_processor_returns_503():
    client = await make_client(secret="correct", processor=None)
    try:
        resp = await client.post(
            "/webhook/telegram",
            json=update_payload(),
            headers={"X-Telegram-Bot-Api-Secret-Token": "correct"},
        )
        assert resp.status_code == 503
    finally:
        await client.__aexit__(None, None, None)
