"""Normalizes raw aiogram Updates into `NormalizedMessage` values.

Handles text, voice, photo, and document messages; ignores edited messages,
messages from the bot itself, and any other update type for now.
"""

from __future__ import annotations

from aiogram import types

from app.interfaces.telegram.normalized import NormalizedMessage


def normalize_update(
    update: types.Update, *, correlation_id: str, source: str = "polling"
) -> NormalizedMessage | None:
    # Silence type-check confusion when only one branch is used.
    msg: types.Message | None = update.message
    if msg is None:
        return None

    sender = msg.from_user
    if sender is None:
        return None
    if sender.is_bot:
        return None

    text: str | None = msg.text or msg.caption
    media_type: str | None = None
    media_file_id: str | None = None
    media_mime: str | None = None
    media_size: int | None = None
    media_caption: str | None = None

    if msg.voice is not None:
        media_type = "voice"
        media_file_id = msg.voice.file_id
        media_mime = msg.voice.mime_type
        media_size = msg.voice.file_size
        media_caption = msg.caption
        text = None
    elif msg.photo:
        largest = max(msg.photo, key=lambda p: p.file_size or 0)
        media_type = "image"
        media_file_id = largest.file_id
        media_mime = "image/jpeg"
        media_size = largest.file_size
        media_caption = msg.caption
        text = None
    elif msg.document is not None:
        media_type = "document"
        media_file_id = msg.document.file_id
        media_mime = msg.document.mime_type
        media_size = msg.document.file_size
        media_caption = msg.caption
        text = None
    elif text is None:
        # Animated stickers/video notes etc. are out of scope for now.
        return None

    return NormalizedMessage(
        correlation_id=correlation_id,
        telegram_user_id=sender.id,
        telegram_username=sender.username,
        telegram_first_name=sender.first_name,
        telegram_last_name=sender.last_name,
        chat_id=msg.chat.id,
        update_id=update.update_id,
        message_id=msg.message_id,
        source=source,
        text=text,
        media_type=media_type,
        media_file_id=media_file_id,
        media_mime_type=media_mime,
        media_file_size=media_size,
        media_caption=media_caption,
        received_at=msg.date,
    )
