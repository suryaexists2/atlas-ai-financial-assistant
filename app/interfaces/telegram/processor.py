"""Telegram update processor: dedupe -> normalize -> persist -> respond.

Purely inbound: it consumes raw updates (from webhook or polling), applies
update_id/message_id deduplication via the ingest ledger, normalizes to a
provider-agnostic NormalizedMessage, persists it, and enqueues any reply
through the outbox. It never talks to the Telegram API directly.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from aiogram import types

from app.application import conversation as conversation_service
from app.application.ingestion.types import MediaIngestionResult
from app.core.logging import get_logger
from app.domain.enums import DocumentStatus, MessageRole
from app.infrastructure.db.session import async_sessionmaker
from app.infrastructure.db.uow import UnitOfWork
from app.interfaces.telegram.normalized import NormalizedMessage
from app.interfaces.telegram.normalizer import normalize_update
from app.interfaces.telegram.sanitize import sanitize_reply

logger = get_logger(__name__)


@dataclass
class ReplyContext:
    """Everything a reply composer needs for one turn."""

    message: NormalizedMessage
    uow: UnitOfWork
    user_id: uuid.UUID
    conversation_id: uuid.UUID
    # Mutable diagnostics the composer can attach (never surfaced to the user,
    # but persisted with the outbox payload for operators).
    note: dict[str, Any] = field(default_factory=dict)
    # Contextual, single-purpose inline button (only used for OAuth connect).
    reply_markup: dict[str, Any] | None = None
    # When the composer already persisted its own assistant message (e.g. a
    # /reset confirmation into a fresh conversation), the processor must not
    # write a second copy into the (possibly deleted) incoming conversation.
    assistant_persisted: bool = False


ReplyComposer = Callable[[ReplyContext], Awaitable[str | None]]

# Turns a media message into a bounded extraction the reply layer can use.
MediaIngestor = Callable[[NormalizedMessage], Awaitable[MediaIngestionResult]]

_KIND_LABEL = {
    "voice": "[voice transcript]",
    "image": "[image contents]",
    "document": "[document contents]",
}


# Context-aware temporary status shown while the reply is being composed.
_STATUS_GOOGLE_RE = re.compile(r"(?i)\b(?:google|gmail|calendar|drive|sheet|meeting|email)\b")
_STATUS_MARKET_RE = re.compile(
    r"(?i)\b(?:quote|price|stock|stocks|market|trading|ticker|earnings|filing|filings|"
    r"news|index|indices|nifty|sensex)\b|\$"
)


def status_text_for(message: NormalizedMessage) -> str:
    """Pick the status bubble text based on what the user just sent."""
    if message.is_media:
        if message.media_type == "voice":
            return "🎙️ Transcribing your voice note..."
        if message.media_type == "document":
            return "📄 Analyzing your document..."
        return "🔎 Looking that up..."
    text = message.combined_text or ""
    if _STATUS_GOOGLE_RE.search(text):
        return "🔗 Checking your connected Google account..."
    if _STATUS_MARKET_RE.search(text):
        return "🔎 Checking the latest market data..."
    return "⏳ Atlas is thinking..."


class UpdateProcessor:
    def __init__(
        self,
        session_factory: async_sessionmaker,
        reply_composer: ReplyComposer,
        *,
        echo_mode: bool = True,
        fallback_reply: str = "Sorry — I hit a temporary hiccup. Give me a moment and try again.",
        media_ingestor: MediaIngestor | None = None,
        status_enabled: bool = True,
    ) -> None:
        self._session_factory = session_factory
        self._reply_composer = reply_composer
        self._echo_mode = echo_mode
        self._fallback_reply = fallback_reply
        self._media_ingestor = media_ingestor
        self._status_enabled = status_enabled

    async def process_update(
        self,
        payload: dict[str, Any],
        *,
        source: str,
        correlation_id: str,
        background_reply: bool = False,
    ) -> bool:
        """Handles one update. True when processed, False when skipped/duplicate.

        When `background_reply` is set, the reply is composed on a fresh session
        in a background task so the webhook can ACK Telegram immediately instead
        of blocking on the LLM turn.
        """
        try:
            update = types.Update.model_validate(payload)
        except Exception:  # noqa: BLE001 - invalid/odd payloads must not crash the webhook
            logger.warning("update_validation_failed", update_id=payload.get("update_id"))
            return False

        if update.edited_message is not None or update.message is None:
            logger.info("update_skipped_non_message", update_id=update.update_id)
            return False

        uow = UnitOfWork(self._session_factory)
        async with uow:
            message = update.message

            # 1) Dedupe (update_id + chat/message) BEFORE anything else.
            inserted = await uow.ingest.record(
                update_id=update.update_id,
                chat_id=message.chat.id,
                message_id=message.message_id,
                source=source,
                correlation_id=correlation_id,
            )
            if not inserted:
                logger.info("update_duplicate_dropped", update_id=update.update_id)
                return False

            # 2) Normalize to a provider-agnostic model.
            normalized = normalize_update(update, correlation_id=correlation_id, source=source)
            if normalized is None:
                logger.info("update_ignored_unrecognized", update_id=update.update_id)
                return False

            # 3) Persist: ensure user + active conversation, store the message.
            user_id: uuid.UUID = await conversation_service.ensure_user(uow, normalized)
            conversation_id: uuid.UUID = (
                await conversation_service.get_or_create_active_conversation(uow, user_id)
            )
            message_id = await conversation_service.persist_incoming_message(
                uow, conversation_id, normalized
            )
            await uow.commit()

        # 4) Reply through the outbox (never direct to Telegram here).
        if not self._echo_mode:
            return True

        if background_reply:
            asyncio.create_task(
                self._compose_and_enqueue(normalized, user_id, conversation_id, message_id),
                name=f"reply:{correlation_id[:12]}",
            )
        else:
            async with UnitOfWork(self._session_factory) as reply_uow:
                await self._maybe_reply(reply_uow, normalized, user_id, conversation_id, message_id)
        return True

    async def _compose_and_enqueue(
        self,
        message: NormalizedMessage,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        message_id: uuid.UUID,
    ) -> None:
        """Background reply path: compose on a fresh session, then persist only the
        outbound message (the LLM turn can be slow and must not hold the webhook)."""
        async with UnitOfWork(self._session_factory) as uow:
            await self._maybe_reply(uow, message, user_id, conversation_id, message_id)

    async def _maybe_reply(
        self,
        uow: UnitOfWork,
        message: NormalizedMessage,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        message_id: uuid.UUID | None = None,
    ) -> None:
        if not self._echo_mode:
            return
        if self._status_enabled:
            # Temporary "thinking" bubble, delivered by the outbox worker before
            # the real reply lands (and deleted right before it is sent).
            await uow.outbox.enqueue(
                chat_id=message.chat_id,
                payload={
                    "type": "status",
                    "correlation_id": message.correlation_id,
                    "text": status_text_for(message),
                },
                priority=100,
            )
        if message.is_media and message_id is not None:
            await self._ingest_media(uow, message, message_id)
        reply_text: str | None = None
        ctx = ReplyContext(
            message=message,
            uow=uow,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        try:
            reply_text = await self._reply_composer(ctx)
        except Exception:  # noqa: BLE001 - never let composer failure break ingestion
            logger.exception("reply_composer_failed", update_id=message.update_id)

        reply_text = reply_text or self._fallback_reply
        reply_text = sanitize_reply(reply_text)
        if reply_text:
            payload: dict[str, Any] = {
                "type": "text",
                "text": reply_text,
                "correlation_id": message.correlation_id,
            }
            if ctx.note:
                payload["debug"] = ctx.note
            if ctx.reply_markup:
                payload["reply_markup"] = ctx.reply_markup
            await uow.outbox.enqueue(
                chat_id=message.chat_id,
                payload=payload,
                priority=10,
            )
            # Persist real replies into the conversation so the LLM context keeps a
            # proper user->assistant->user rhythm. Fallback replies are transient
            # errors and are not worth polluting the memory with. Reset replies
            # are persisted by the composer into the fresh conversation.
            if reply_text != self._fallback_reply and not ctx.assistant_persisted:
                await uow.conversations.add_message(
                    conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=reply_text,
                    correlation_id=message.correlation_id,
                )
            logger.info("reply_enqueued", update_id=message.update_id)

    async def _ingest_media(
        self,
        uow: UnitOfWork,
        message: NormalizedMessage,
        message_id: uuid.UUID,
    ) -> None:
        """Downloads, parses, persists the document, and enriches the stored
        user message so the agent sees the extracted content.

        Failures are never silent: an extraction error or an unexpected
        exception leaves a `documents` row (status FAILED) and an entry in the
        message's `media_meta`, so operators can see exactly what happened and
        the reply path still runs gracefully.
        """
        user = await uow.users.get_by_telegram_id(message.telegram_user_id)
        if user is None or self._media_ingestor is None:
            return

        base_meta: dict[str, Any] = {
            "file_id": message.media_file_id,
            "mime_type": message.media_mime_type,
            "filename": message.media_filename,
            "kind": message.media_type,
            "download_size": message.media_file_size,
        }
        try:
            result = await self._media_ingestor(message)
        except Exception as exc:  # noqa: BLE001 - ingestion must never crash the reply path
            logger.exception("media_ingest_failed", media_type=message.media_type)
            base_meta.update(error="ingestion raised unexpectedly", error_code="internal")
            base_meta["error_detail"] = str(exc)[:500]
            await self._fail_document(uow, user.id, base_meta, message, message_id)
            return

        if result.error is not None:
            base_meta.update(error=result.error, error_code=result.error_code)
            await self._fail_document(uow, user.id, base_meta, message, message_id)
            return

        document = result.document
        if document is None:
            base_meta.update(error="nothing was extracted", error_code="empty")
            await self._fail_document(uow, user.id, base_meta, message, message_id)
            return

        kind = document.kind.value
        base_meta.update(kind=kind, chunk_count=document.chunk_count, truncated=document.truncated)
        document_row = await self._doc_record(uow, user.id, base_meta, message)
        await uow.documents.update_status(
            document_row,
            DocumentStatus.PROCESSED,
            **{
                "kind": kind,
                "extracted_text": document.text,
                "chunk_count": document.chunk_count,
                "truncated": document.truncated,
            },
        )

        display = result.content if result.content else document.text
        label = _KIND_LABEL.get(message.media_type or kind, "[contents]")
        if message.media_caption:
            content = f"{message.media_caption}\n\n{label}\n{display}"
        else:
            content = f"{label}\n{display}"
        await uow.conversations.update_message(
            message_id,
            content=content,
            media_meta={
                "kind": kind,
                "chunk_count": document.chunk_count,
                "truncated": document.truncated,
                "excerpt": display,
            },
        )
        await uow.commit()
        logger.info(
            "media_ingested_and_linked",
            media_type=message.media_type,
            kind=kind,
            chars=len(display),
        )

    async def _fail_document(
        self,
        uow: UnitOfWork,
        user_id: uuid.UUID,
        base_meta: dict[str, Any],
        message: NormalizedMessage,
        message_id: uuid.UUID,
    ) -> None:
        """Records a failed ingestion so it is never invisible."""
        document_row = await self._doc_record(uow, user_id, base_meta, message)
        await uow.documents.update_status(
            document_row,
            DocumentStatus.FAILED,
            **{"error": base_meta.get("error"), "error_code": base_meta.get("error_code")},
        )
        await uow.conversations.update_message(
            message_id,
            media_meta={
                "error": base_meta.get("error"),
                "error_code": base_meta.get("error_code"),
                "kind": base_meta.get("kind"),
            },
        )
        await uow.commit()

    async def _doc_record(
        self,
        uow: UnitOfWork,
        user_id: uuid.UUID,
        meta: dict[str, Any],
        message: NormalizedMessage,
    ):
        """Creates a documents row (one per media message)."""
        return await uow.documents.create(
            user_id,
            filename=meta.get("filename") or message.media_type or "attachment",
            mime_type=meta.get("mime_type"),
            size_bytes=meta.get("download_size"),
            status=DocumentStatus.PENDING,
            doc_meta={**meta, "conversation_id": str(message.correlation_id)},
        )
