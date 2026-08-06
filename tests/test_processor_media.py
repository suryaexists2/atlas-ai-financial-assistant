"""UpdateProcessor + media ingestion integration tests."""

from sqlalchemy import select

from app.application.ingestion.types import MediaIngestionResult, ParsedDocument
from app.domain.entities import OutboundMessage
from app.domain.enums import DocumentKind, DocumentStatus
from app.infrastructure.db.uow import UnitOfWork
from app.interfaces.telegram.processor import UpdateProcessor
from tests.conftest import tg_text_update, tg_voice_update


def make_result(
    *,
    text="what is the revenue growth",
    error=None,
    error_code=None,
    kind=DocumentKind.VOICE,
):
    doc = (
        ParsedDocument(kind=kind, text=text, filename="note.ogg", mime_type="audio/ogg")
        if error is None
        else None
    )
    return MediaIngestionResult(
        document=doc,
        content=text if error is None else None,
        error=error,
        error_code=error_code,
    )


async def compose(ctx) -> str:
    return f"echo:{ctx.message.media_type} ok"


async def test_media_success_enriches_message_and_documents(session_factory):
    calls = []

    async def ingestor(msg):
        calls.append(msg.media_file_id)
        return make_result()

    processor = UpdateProcessor(session_factory, compose, media_ingestor=ingestor)
    payload = tg_voice_update(chat_id=888, message_id=70, file_id="f-voice-1")
    assert await processor.process_update(payload, source="webhook", correlation_id="m-1") is True

    uow = UnitOfWork(session_factory)
    async with uow:
        user = await uow.users.get_by_telegram_id(888)
        convos = await uow.conversations.list_for_user(user.id)
        messages = await uow.conversations.list_messages(convos[0].id)
        user_msg = messages[0]
        assert "[voice transcript]" in user_msg.content
        assert "what is the revenue" in user_msg.content
        assert user_msg.media_meta["kind"] == "voice"
        assert user_msg.media_meta["file_id"] == "f-voice-1"

        docs = await uow.documents.list_for_user(user.id)
        assert len(docs) == 1
        assert docs[0].status is DocumentStatus.PROCESSED
        assert docs[0].doc_meta["kind"] == "voice"
        assert "what is the revenue" in docs[0].doc_meta["extracted_text"]
    assert calls == ["f-voice-1"]


async def test_media_failure_marks_document_failed(session_factory):
    async def ingestor(msg):
        return MediaIngestionResult(error="Sorry, could not transcribe.", error_code="stt")

    processor = UpdateProcessor(session_factory, compose, media_ingestor=ingestor)
    payload = tg_voice_update(chat_id=888, message_id=71, file_id="f-voice-2")
    assert await processor.process_update(payload, source="webhook", correlation_id="m-2") is True

    uow = UnitOfWork(session_factory)
    async with uow:
        user = await uow.users.get_by_telegram_id(888)
        docs = await uow.documents.list_for_user(user.id)
        assert len(docs) == 1
        assert docs[0].status is DocumentStatus.FAILED
        assert docs[0].doc_meta["error_code"] == "stt"
        convos = await uow.conversations.list_for_user(user.id)
        messages = await uow.conversations.list_messages(convos[0].id)
        assert messages[0].media_meta["error_code"] == "stt"


async def test_media_ingestor_exception_marks_document_failed(session_factory):
    async def ingestor(msg):
        raise RuntimeError("boom")

    processor = UpdateProcessor(session_factory, compose, media_ingestor=ingestor)
    payload = tg_voice_update(chat_id=888, message_id=101, file_id="f3")
    processed = await processor.process_update(payload, source="webhook", correlation_id="m-3")
    assert processed is True

    uow = UnitOfWork(session_factory)
    async with uow:
        result = await uow.session.execute(select(OutboundMessage))
        assert len(result.scalars().all()) == 1
        user = await uow.users.get_by_telegram_id(888)
        docs = await uow.documents.list_for_user(user.id)
        assert len(docs) == 1
        assert docs[0].status is DocumentStatus.FAILED
        assert docs[0].doc_meta["error_code"] == "internal"


async def test_plain_text_never_calls_ingestor(session_factory):
    called = False

    async def ingestor(msg):
        nonlocal called
        called = True
        return make_result()

    processor = UpdateProcessor(session_factory, compose, media_ingestor=ingestor)
    payload = tg_text_update(update_id=2, chat_id=888, message_id=102, text="hello text")
    await processor.process_update(payload, source="webhook", correlation_id="m-4")
    assert called is False


async def test_media_caption_plus_transcript_both_visible(session_factory):
    payload = tg_voice_update(chat_id=888, message_id=103, file_id="f-voice-3")
    payload["message"]["caption"] = "answer fast please"

    async def ingestor(msg):
        return make_result(text="compare TSLA and NVDA")

    processor = UpdateProcessor(session_factory, compose, media_ingestor=ingestor)
    await processor.process_update(payload, source="webhook", correlation_id="m-5")

    uow = UnitOfWork(session_factory)
    async with uow:
        user = await uow.users.get_by_telegram_id(888)
        convos = await uow.conversations.list_for_user(user.id)
        messages = await uow.conversations.list_messages(convos[0].id)
        content = messages[0].content
        assert "answer fast please" in content
        assert "compare TSLA and NVDA" in content
