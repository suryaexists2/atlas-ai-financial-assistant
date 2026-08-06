"""Application services for conversations (used by the Telegram entry layer)."""

from __future__ import annotations

import datetime as dt
import uuid

from app.domain.enums import MessageRole
from app.infrastructure.db.uow import UnitOfWork
from app.interfaces.telegram.normalized import NormalizedMessage

_ACTIVE_CONVERSATION_WINDOW = dt.timedelta(hours=24)


async def ensure_user(uow: UnitOfWork, message: NormalizedMessage) -> uuid.UUID:
    """Gets or creates the internal user for a Telegram sender."""
    user = await uow.users.get_by_telegram_id(message.telegram_user_id)
    if user is None:
        user = await uow.users.create(
            telegram_id=message.telegram_user_id,
            username=message.telegram_username,
            first_name=message.telegram_first_name,
            last_name=message.telegram_last_name,
        )
    else:
        changed: dict = {}
        if message.telegram_username is not None and user.username != message.telegram_username:
            changed["username"] = message.telegram_username
        if (
            message.telegram_first_name is not None
            and user.first_name != message.telegram_first_name
        ):
            changed["first_name"] = message.telegram_first_name
        if changed:
            await uow.users.update(user, **changed)
    return user.id


async def get_or_create_active_conversation(uow: UnitOfWork, user_id: uuid.UUID) -> uuid.UUID:
    """Reuses the most recent conversation if recent; otherwise opens a new one."""
    conversations = await uow.conversations.list_for_user(user_id, limit=1)
    if conversations:
        latest = conversations[0]
        window = dt.datetime.now(dt.UTC) - _ACTIVE_CONVERSATION_WINDOW
        if latest.created_at.replace(tzinfo=dt.UTC) > window:
            return latest.id
    conversation = await uow.conversations.create(user_id)
    return conversation.id


async def persist_incoming_message(
    uow: UnitOfWork, conversation_id: uuid.UUID, message: NormalizedMessage
) -> None:
    """Stores the normalized message in the conversation history."""
    from app.domain.enums import ContentType

    content_type = ContentType(message.media_type) if message.is_media else ContentType.TEXT
    await uow.conversations.add_message(
        conversation_id,
        role=MessageRole.USER,
        content=message.combined_text,
        content_type=content_type,
        media_meta=(
            {
                "file_id": message.media_file_id,
                "mime_type": message.media_mime_type,
                "file_size": message.media_file_size,
                "caption": message.media_caption,
            }
            if message.is_media
            else None
        ),
        correlation_id=message.correlation_id,
    )
