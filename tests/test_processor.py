"""UpdateProcessor end-to-end tests: dedup, persistence, outbox enqueue."""

import pytest
from sqlalchemy import func, select

from app.domain.entities import OutboundMessage
from app.infrastructure.db.uow import UnitOfWork
from app.interfaces.telegram.processor import UpdateProcessor
from tests.conftest import tg_text_update, tg_voice_update


async def _compose(message) -> str:
    return f"echo:{message.combined_text or message.media_type}"


def make_processor(session_factory, composer=_compose) -> UpdateProcessor:
    return UpdateProcessor(session_factory, composer, echo_mode=True)


@pytest.mark.asyncio
async def test_text_message_processed_end_to_end(session_factory):
    processor = make_processor(session_factory)
    payload = tg_text_update(update_id=1, chat_id=777, message_id=21, text="hello")

    processed = await processor.process_update(payload, source="webhook", correlation_id="c-1")
    assert processed is True

    uow = UnitOfWork(session_factory)
    async with uow:
        user = await uow.users.get_by_telegram_id(777)
        assert user is not None
        convos = await uow.conversations.list_for_user(user.id)
        assert len(convos) == 1
        messages = await uow.conversations.list_messages(convos[0].id)
        assert len(messages) == 1
        assert messages[0].content == "hello"
        assert messages[0].correlation_id == "c-1"

        result = await uow.session.execute(select(OutboundMessage))
        outbound = result.scalars().all()
        assert len(outbound) == 1
        assert outbound[0].payload["text"] == "echo:hello"
        assert outbound[0].priority == 10


@pytest.mark.asyncio
async def test_duplicate_update_id_is_dropped(session_factory):
    processor = make_processor(session_factory)
    payload = tg_text_update(update_id=5, chat_id=777, message_id=22, text="once")

    assert await processor.process_update(payload, source="webhook", correlation_id="c-2") is True
    assert await processor.process_update(payload, source="webhook", correlation_id="c-2b") is False

    uow = UnitOfWork(session_factory)
    async with uow:
        user = await uow.users.get_by_telegram_id(777)
        convos = await uow.conversations.list_for_user(user.id)
        messages = await uow.conversations.list_messages(convos[0].id)
        assert len(messages) == 1  # stored exactly once
        result = await uow.session.execute(select(func.count()).select_from(OutboundMessage))
        assert result.scalar_one() == 1  # replied exactly once


@pytest.mark.asyncio
async def test_same_message_redelivered_with_new_update_id_is_dropped(session_factory):
    processor = make_processor(session_factory)
    first = tg_text_update(update_id=10, chat_id=777, message_id=30, text="dup")
    replay = tg_text_update(update_id=11, chat_id=777, message_id=30, text="dup")

    assert await processor.process_update(first, source="webhook", correlation_id="c-3") is True
    assert await processor.process_update(replay, source="webhook", correlation_id="c-3b") is False


@pytest.mark.asyncio
async def test_voice_message_persisted_as_media(session_factory):
    processor = make_processor(session_factory)
    payload = tg_voice_update(chat_id=777, message_id=31)

    assert await processor.process_update(payload, source="polling", correlation_id="c-4") is True

    uow = UnitOfWork(session_factory)
    async with uow:
        user = await uow.users.get_by_telegram_id(777)
        convos = await uow.conversations.list_for_user(user.id)
        messages = await uow.conversations.list_messages(convos[0].id)
        assert messages[0].content is None
        assert messages[0].media_meta["file_id"] == "f123"


@pytest.mark.asyncio
async def test_edited_update_never_reaches_processor(session_factory):
    processor = make_processor(session_factory)
    payload = tg_text_update(update_id=20, chat_id=777, message_id=40, text="edited")
    payload["edited_message"] = payload.pop("message")

    assert await processor.process_update(payload, source="webhook", correlation_id="c-5") is False


@pytest.mark.asyncio
async def test_echo_disabled_still_persists(session_factory):
    processor = UpdateProcessor(session_factory, _compose, echo_mode=False)
    payload = tg_text_update(update_id=30, chat_id=777, message_id=50, text="quiet")

    assert await processor.process_update(payload, source="webhook", correlation_id="c-6") is True

    uow = UnitOfWork(session_factory)
    async with uow:
        result = await uow.session.execute(select(func.count()).select_from(OutboundMessage))
        assert result.scalar_one() == 0  # no reply enqueued


@pytest.mark.asyncio
async def test_composer_failure_does_not_break_ingestion(session_factory):
    async def broken_composer(message):
        raise RuntimeError("composer exploded")

    processor = make_processor(session_factory, composer=broken_composer)
    payload = tg_text_update(update_id=40, chat_id=777, message_id=60, text="still stored")

    assert await processor.process_update(payload, source="webhook", correlation_id="c-7") is True

    uow = UnitOfWork(session_factory)
    async with uow:
        user = await uow.users.get_by_telegram_id(777)
        convos = await uow.conversations.list_for_user(user.id)
        assert len(await uow.conversations.list_messages(convos[0].id)) == 1
