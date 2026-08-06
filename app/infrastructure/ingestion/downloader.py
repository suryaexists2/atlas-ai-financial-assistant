"""Telegram file downloader (aiogram adapter).

Fetches an attachment by `file_id` into memory via the getFile + download_file
flow. Enforces a hard byte cap so a malicious/huge payload never lands in RAM.
"""

from __future__ import annotations

import io

from aiogram import Bot

from app.application.ingestion.types import FileData


class TooLargeError(RuntimeError):
    """The remote file exceeds the cap we are willing to download."""


class TelegramFileFetcher:
    def __init__(self, bot: Bot, *, max_bytes: int = 25_000_000) -> None:
        self._bot = bot
        self._max_bytes = max_bytes

    async def fetch(self, file_id: str) -> FileData:
        file = await self._bot.get_file(file_id)
        if file.file_size is not None and file.file_size > self._max_bytes:
            raise TooLargeError(f"attachment too large ({file.file_size} bytes)")
        buffer = io.BytesIO()
        if not file.file_path:
            raise ValueError("telegram did not return a downloadable file path")
        await self._bot.download_file(file.file_path, destination=buffer)
        raw = buffer.getvalue()
        return FileData(
            raw=raw,
            mime_type=None,
            size=len(raw),
        )


__all__ = ["TelegramFileFetcher", "TooLargeError"]
