"""UpdateProcessor end-to-end tests: dedup, persistence, outbox enqueue."""

import pytest
from sqlalchemy import func, select

from app.domain.entities import OutboundMessage
from app.infrastructure.db.uow import UnitOfWork
from app.interfaces.telegram.processor import UpdateProcessor
from app.interfaces.telegram.sanitize import sanitize_reply
from tests.conftest import tg_text_update, tg_voice_update


async def _compose(ctx) -> str:
    message = ctx.message
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
        assert len(messages) == 2  # user msg + persisted assistant reply
        assert messages[0].content == "hello"
        assert messages[0].correlation_id == "c-1"
        assert messages[1].role.value == "assistant"
        assert messages[1].content == "echo:hello"

        result = await uow.session.execute(select(OutboundMessage))
        outbound = result.scalars().all()
        assert len(outbound) == 2  # status bubble + final reply
        text_rows = [o for o in outbound if o.payload.get("type") != "status"]
        status_rows = [o for o in outbound if o.payload.get("type") == "status"]
        assert len(text_rows) == 1
        assert len(status_rows) == 1
        assert text_rows[0].payload["text"] == "echo:hello"
        assert text_rows[0].priority == 10
        assert status_rows[0].priority == 100
        assert status_rows[0].payload["correlation_id"] == "c-1"


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
        assert len(messages) == 2  # user + assistant reply, both stored once
        result = await uow.session.execute(select(func.count()).select_from(OutboundMessage))
        assert result.scalar_one() == 2  # status bubble + reply, each exactly once


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
async def test_fallback_reply_not_persisted_to_conversation(session_factory):
    async def fallback_composer(ctx):
        return "FALLBACK"

    processor = UpdateProcessor(session_factory, fallback_composer, fallback_reply="FALLBACK")
    payload = tg_text_update(update_id=32, chat_id=777, message_id=32, text="hello")

    assert await processor.process_update(payload, source="webhook", correlation_id="c-4b") is True

    uow = UnitOfWork(session_factory)
    async with uow:
        user = await uow.users.get_by_telegram_id(777)
        convos = await uow.conversations.list_for_user(user.id)
        messages = await uow.conversations.list_messages(convos[0].id)
        roles = [m.role.value for m in messages]
        assert roles == ["user"]  # fallback must not pollute context


@pytest.mark.asyncio
async def test_invalid_payload_returns_false(session_factory):
    processor = make_processor(session_factory)
    bad = tg_voice_update(chat_id=777, message_id=99)
    del bad["message"]["voice"]["file_unique_id"]  # make it invalid per aiogram schema

    assert await processor.process_update(bad, source="webhook", correlation_id="c-9") is False

    uow = UnitOfWork(session_factory)
    async with uow:
        result = await uow.session.execute(select(func.count()).select_from(OutboundMessage))
        assert result.scalar_one() == 0


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
async def test_composer_failure_still_enqueues_fallback(session_factory):
    async def broken_composer(ctx):
        raise RuntimeError("composer exploded")

    processor = make_processor(session_factory, composer=broken_composer)
    payload = tg_text_update(update_id=40, chat_id=777, message_id=60, text="still stored")

    assert await processor.process_update(payload, source="webhook", correlation_id="c-7") is True

    uow = UnitOfWork(session_factory)
    async with uow:
        user = await uow.users.get_by_telegram_id(777)
        convos = await uow.conversations.list_for_user(user.id)
        assert len(await uow.conversations.list_messages(convos[0].id)) == 1
        result = await uow.session.execute(select(OutboundMessage))
        outbound = result.scalars().all()
        text_rows = [o for o in outbound if o.payload.get("type") != "status"]
        assert len(text_rows) == 1
        assert (
            text_rows[0].payload["text"]
            == UpdateProcessor(session_factory, _compose)._fallback_reply
        )


@pytest.mark.asyncio
async def test_composer_none_still_enqueues_fallback(session_factory):
    async def silent_composer(ctx):
        return None

    processor = UpdateProcessor(session_factory, silent_composer, fallback_reply="custom fallback")
    payload = tg_text_update(update_id=41, chat_id=777, message_id=61, text="quiet")

    assert await processor.process_update(payload, source="webhook", correlation_id="c-8") is True

    uow = UnitOfWork(session_factory)
    async with uow:
        result = await uow.session.execute(select(OutboundMessage))
        outbound = result.scalars().all()
        text_rows = [o for o in outbound if o.payload.get("type") != "status"]
        assert len(text_rows) == 1
        assert text_rows[0].payload["text"] == "custom fallback"


def test_leaked_tool_names_stripped_from_reply():
    assert sanitize_reply("Let me check. (get_market_quote) Here is the price") == (
        "Let me check. Here is the price"
    )
    assert sanitize_reply("(get_market_news) He") == "He"
    assert sanitize_reply("Plain text (AAPL) stays (not a tool)") == (
        "Plain text (AAPL) stays (not a tool)"
    )


@pytest.mark.asyncio
async def test_leaky_composer_reply_is_sanitized_before_outbox(session_factory):
    async def leaky_composer(ctx):
        return "Nvidia is at (get_market_quote(symbol=\"NVDA\")) $218.99 today."

    processor = UpdateProcessor(session_factory, leaky_composer, echo_mode=True)
    payload = tg_text_update(update_id=99, chat_id=777, message_id=71, text="nvda?")

    assert await processor.process_update(payload, source="webhook", correlation_id="c-9") is True

    uow = UnitOfWork(session_factory)
    async with uow:
        result = await uow.session.execute(select(OutboundMessage))
        outbound = result.scalars().all()
        text_rows = [o for o in outbound if o.payload.get("type") != "status"]
        assert len(text_rows) == 1
        assert "get_market_quote" not in text_rows[0].payload["text"]
        assert text_rows[0].payload["text"] == "Nvidia is at $218.99 today."


def test_status_text_is_context_aware():
    from app.interfaces.telegram.normalized import NormalizedMessage
    from app.interfaces.telegram.processor import status_text_for

    def msg(text=None, media_type=None):
        return NormalizedMessage(
            update_id=1,
            chat_id=1,
            telegram_user_id=1,
            message_id=1,
            text=text,
            media_type=media_type,
            correlation_id="c",
            source="test",
        )

    assert status_text_for(msg("What is NVDA trading at?")) == (
        "🔎 Checking the latest market data..."
    )
    assert status_text_for(msg("connect my gmail")) == (
        "🔗 Checking your connected Google account..."
    )
    assert status_text_for(msg("hi")) == "⏳ Atlas is thinking..."
    assert status_text_for(msg(media_type="voice")) == "🎙️ Transcribing your voice note..."
    assert status_text_for(msg(media_type="document")) == "📄 Analyzing your document..."
    assert status_text_for(msg(media_type="image")) == "🔎 Looking that up..."


@pytest.mark.asyncio
async def test_status_enqueued_before_final_and_disabled_flag(session_factory):
    processor = make_processor(session_factory)
    payload = tg_text_update(update_id=88, chat_id=777, message_id=81, text="TSLA price?")
    await processor.process_update(payload, source="webhook", correlation_id="c-10")

    uow = UnitOfWork(session_factory)
    async with uow:
        result = await uow.session.execute(select(OutboundMessage))
        rows = result.scalars().all()
        statuses = [r for r in rows if r.payload.get("type") == "status"]
        assert len(statuses) == 1
        assert statuses[0].priority == 100
        assert statuses[0].created_at <= rows[-1].created_at

    quiet = UpdateProcessor(session_factory, _compose, status_enabled=False)
    await quiet.process_update(
        tg_text_update(update_id=89, chat_id=777, message_id=82, text="hi"),
        source="webhook",
        correlation_id="c-11",
    )
    uow = UnitOfWork(session_factory)
    async with uow:
        result = await uow.session.execute(select(OutboundMessage))
        rows = result.scalars().all()
        statuses = [r for r in rows if r.payload.get("type") == "status"]
        assert len(statuses) == 1  # only the c-10 one
