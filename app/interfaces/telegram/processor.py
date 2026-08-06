"""Telegram update processor: dedupe -> normalize -> persist -> respond.

Purely inbound: it consumes raw updates (from webhook or polling), applies
update_id/message_id deduplication via the ingest ledger, normalizes to a
provider-agnostic NormalizedMessage, persists it, and enqueues any reply
through the outbox. It never talks to the Telegram API directly.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from aiogram import types

from app.application import conversation as conversation_service
from app.core.logging import get_logger
from app.infrastructure.db.session import async_sessionmaker
from app.infrastructure.db.uow import UnitOfWork
from app.interfaces.telegram.normalized import NormalizedMessage
from app.interfaces.telegram.normalizer import normalize_update

logger = get_logger(__name__)


@dataclass
class ReplyContext:
    """Everything a reply composer needs for one turn."""

    message: NormalizedMessage
    uow: UnitOfWork
    user_id: uuid.UUID
    conversation_id: uuid.UUID


ReplyComposer = Callable[[ReplyContext], Awaitable[str | None]]


class UpdateProcessor:
    def __init__(
        self,
        session_factory: async_sessionmaker,
        reply_composer: ReplyComposer,
        *,
        echo_mode: bool = True,
    ) -> None:
        self._session_factory = session_factory
        self._reply_composer = reply_composer
        self._echo_mode = echo_mode

    async def process_update(
        self, payload: dict[str, Any], *, source: str, correlation_id: str
    ) -> bool:
        """Handles one update. True when processed, False when skipped/duplicate."""
        update = types.Update.model_validate(payload)

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
            await conversation_service.persist_incoming_message(uow, conversation_id, normalized)

            # 4) Reply through the outbox (never direct to Telegram here).
            await self._maybe_reply(uow, normalized, user_id, conversation_id)
        return True

    async def _maybe_reply(
        self,
        uow: UnitOfWork,
        message: NormalizedMessage,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
    ) -> None:
        if not self._echo_mode:
            return
        reply_text: str | None = None
        try:
            reply_text = await self._reply_composer(
                ReplyContext(
                    message=message,
                    uow=uow,
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
            )
        except Exception:  # noqa: BLE001 - never let composer failure break ingestion
            logger.exception("reply_composer_failed", update_id=message.update_id)

        if reply_text:
            await uow.outbox.enqueue(
                chat_id=message.chat_id,
                payload={
                    "type": "text",
                    "text": reply_text,
                    "correlation_id": message.correlation_id,
                },
                priority=10,
            )
            logger.info("reply_enqueued", update_id=message.update_id)
