"""Normalizer tests: aiogram Updates -> provider-agnostic NormalizedMessage."""

import pytest
from aiogram import types

from app.interfaces.telegram.normalized import NormalizedMessage
from app.interfaces.telegram.normalizer import normalize_update
from tests.conftest import tg_photo_update, tg_text_update, tg_voice_update


def _as_update(payload: dict) -> types.Update:
    return types.Update.model_validate(payload)


@pytest.mark.asyncio
async def test_text_message_normalized():
    update = _as_update(tg_text_update(update_id=1, chat_id=555, message_id=10, text="hello"))
    msg = normalize_update(update, correlation_id="corr-1")

    assert isinstance(msg, NormalizedMessage)
    assert msg.text == "hello"
    assert msg.telegram_user_id == 555
    assert msg.chat_id == 555
    assert msg.update_id == 1
    assert msg.message_id == 10
    assert msg.source == "polling"  # normalizer default; caller overrides
    assert msg.media_type is None
    assert msg.combined_text == "hello"


@pytest.mark.asyncio
async def test_voice_message_normalized():
    update = _as_update(tg_voice_update(chat_id=555, message_id=11))
    msg = normalize_update(update, correlation_id="corr-2")

    assert msg.media_type == "voice"
    assert msg.media_file_id == "f123"
    assert msg.media_mime_type == "audio/ogg"
    assert msg.text is None
    assert msg.is_media is True


@pytest.mark.asyncio
async def test_photo_uses_largest_size_and_caption():
    update = _as_update(tg_photo_update(chat_id=555, message_id=12))
    msg = normalize_update(update, correlation_id="corr-3")

    assert msg.media_type == "image"
    assert msg.media_file_id == "ph_big"
    assert msg.media_caption == "check this chart"
    assert msg.combined_text == "check this chart"


@pytest.mark.asyncio
async def test_document_normalized():
    payload = {
        "update_id": 900,
        "message": {
            "message_id": 13,
            "date": 1780000000,
            "from": {"id": 555, "is_bot": False, "first_name": "Tester"},
            "chat": {"id": 555, "type": "private", "first_name": "Tester"},
            "document": {
                "file_id": "doc_1",
                "file_unique_id": "fu_doc",
                "file_name": "annual.pdf",
                "mime_type": "application/pdf",
                "file_size": 12345,
            },
        },
    }
    msg = normalize_update(_as_update(payload), correlation_id="corr-4")

    assert msg.media_type == "document"
    assert msg.media_file_id == "doc_1"
    assert msg.media_mime_type == "application/pdf"


@pytest.mark.asyncio
async def test_edited_message_is_ignored():
    payload = tg_text_update(update_id=1, chat_id=555, message_id=10, text="old")
    payload["edited_message"] = payload.pop("message")
    assert normalize_update(_as_update(payload), correlation_id="corr-5") is None


@pytest.mark.asyncio
async def test_bot_message_is_ignored():
    payload = {
        "update_id": 1,
        "message": {
            "message_id": 14,
            "date": 1780000000,
            "from": {"id": 555, "is_bot": True, "first_name": "Atlas"},
            "chat": {"id": 555, "type": "private", "first_name": "Tester"},
            "text": "I am a bot",
        },
    }
    assert normalize_update(_as_update(payload), correlation_id="corr-6") is None


@pytest.mark.asyncio
async def test_callback_query_is_ignored():
    payload = {
        "update_id": 1,
        "callback_query": {
            "id": "q1",
            "from": {"id": 555, "is_bot": False, "first_name": "Tester"},
            "chat_instance": "12345",
        },
    }
    assert normalize_update(_as_update(payload), correlation_id="corr-7") is None
