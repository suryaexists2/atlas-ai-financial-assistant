"""Provider-agnostic incoming message model.

Every Telegram update (text, voice, image, document) is normalized into this
shape BEFORE it reaches the Agent Core, so the core never depends on aiogram.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field


class NormalizedMessage(BaseModel):
    correlation_id: str
    telegram_user_id: int
    telegram_username: str | None = None
    telegram_first_name: str | None = None
    telegram_last_name: str | None = None
    chat_id: int
    update_id: int
    message_id: int
    source: str  # "webhook" | "polling"
    text: str | None = None
    media_type: str | None = None  # "voice" | "image" | "document"
    media_file_id: str | None = None
    media_mime_type: str | None = None
    media_file_size: int | None = None
    media_filename: str | None = None
    media_caption: str | None = None
    received_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.UTC))

    @property
    def is_media(self) -> bool:
        return self.media_type is not None

    @property
    def combined_text(self) -> str | None:
        """Caption (for media) or the message text, whichever is present."""
        if self.text is not None:
            return self.text
        return self.media_caption
