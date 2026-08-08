"""Pipeline orchestration: fetch -> parse -> chunk -> summarize for context.

The pipeline is constructed once per app process with the real downloader and
parser set, then handed to the Telegram processor. It never touches the DB —
persistence happens one layer up (the processor commits document rows).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.application.ingestion import chunker
from app.application.ingestion.parsers import ParserRegistry
from app.application.ingestion.types import (
    FileData,
    MediaIngestionResult,
    ParsedDocument,
    ParseError,
    UnsupportedMediaError,
)
from app.core.logging import get_logger
from app.domain.enums import DocumentKind

logger = get_logger(__name__)


class FileFetcher(Protocol):
    """Downloads an attachment by provider file id."""

    async def fetch(
        self,
        file_id: str,
        *,
        mime_type: str | None = None,
        filename: str | None = None,
    ) -> FileData: ...


class SpeechToText(Protocol):
    """Transcribes an audio payload into plain text."""

    async def transcribe(self, data: FileData) -> str: ...


class VisionAnalyzer(Protocol):
    """Describes/reads an image payload into plain text (incl. basic OCR)."""

    async def describe(self, data: FileData) -> str: ...


@dataclass
class IngestionPipeline:
    registry: ParserRegistry
    fetcher: FileFetcher
    stt: SpeechToText | None = None
    vision: VisionAnalyzer | None = None
    max_bytes: int = 25_000_000
    max_chars: int = 120_000
    chunk_chars: int = 12_000
    excerpt_chars: int = 8_000

    async def process(
        self,
        *,
        file_id: str,
        mime_type: str | None,
        filename: str | None,
        data: FileData | None = None,
    ) -> MediaIngestionResult:
        """Downloads and parses one attachment, returning a bounded result.

        When `data` is provided (e.g. a Drive download already in memory), the
        fetcher is skipped and the bytes go straight through the same parse /
        chunk path used for Telegram uploads.
        """
        try:
            if data is not None:
                if len(data.raw) > self.max_bytes:
                    return MediaIngestionResult(
                        error="That file is too large for me to read.",
                        error_code="too_large",
                        meta={"filename": filename, "size": len(data.raw)},
                    )
            else:
                data = await self.fetcher.fetch(file_id, mime_type=mime_type, filename=filename)
                if len(data.raw) > self.max_bytes:
                    return MediaIngestionResult(
                        error="That file is too large for me to read.",
                        error_code="too_large",
                        meta={"file_id": file_id, "size": len(data.raw)},
                    )
        except Exception as exc:  # noqa: BLE001 - network/provider failures are graceful
            logger.warning("media_download_failed", file_id=file_id, error=str(exc))
            return MediaIngestionResult(
                error="Sorry, I could not download that file.",
                error_code="download",
                meta={"file_id": file_id},
            )

        # Some fetchers (e.g. Telegram) return raw bytes without mime/filename;
        # classification must fall back to the request-level attributes.
        data = FileData(
            raw=data.raw,
            size=data.size or len(data.raw),
            mime_type=data.mime_type or mime_type,
            filename=data.filename or filename,
        )

        kind = data.kind
        if kind is DocumentKind.UNSUPPORTED:
            return MediaIngestionResult(
                error="I can read text, PDF, spreadsheets, images, and voice notes.",
                error_code="unsupported",
                meta={"file_id": file_id, "mime_type": data.mime_type, "filename": data.filename},
            )

        text: str | None = None
        try:
            if kind is DocumentKind.VOICE:
                text = await self._transcribe(data)
            elif kind is DocumentKind.IMAGE:
                text = await self._analyze_image(data)
            else:
                text = await self._parse_text(data)
        except RuntimeError as exc:
            # Provider-level failures (auth, model gone, empty reply) must not
            # masquerade as "nothing was extracted": record the real reason so
            # operators can see exactly which AI stage and provider failed.
            logger.warning("media_ai_stage_failed", kind=kind.value, error=str(exc))
            return MediaIngestionResult(
                error=f"AI processing failed: {exc}",
                error_code="ai_stage",
                meta={"file_id": file_id, "kind": kind},
            )
        except Exception as exc:  # noqa: BLE001 - AI stage crashes are graceful
            logger.warning("media_ai_stage_crashed", kind=kind.value, error=str(exc))
            return MediaIngestionResult(
                error="AI processing crashed",
                error_code="internal",
                meta={"file_id": file_id, "kind": kind},
            )

        if not text:
            return MediaIngestionResult(
                error="I could not read anything from that file.",
                error_code="empty",
                meta={"file_id": file_id, "kind": kind},
            )

        truncated = len(text) > self.max_chars
        full = text[: self.max_chars]
        excerpt, excerpt_truncated = chunker.truncate_excerpt(full, max_chars=self.excerpt_chars)
        chunks = chunker.chunk_text(full, max_chars=self.chunk_chars)
        document = ParsedDocument(
            kind=kind,
            text=full,
            filename=data.filename or filename,
            mime_type=data.mime_type or mime_type,
            byte_size=len(data.raw),
            chunk_count=len(chunks),
            truncated=truncated or excerpt_truncated,
            meta={
                "chunk_count": len(chunks),
                "full_text": full,
                "note": "preview" if excerpt_truncated else "full",
            },
        )
        logger.info(
            "media_ingested",
            kind=kind.value,
            bytes=len(data.raw),
            chars=len(full),
            chunks=len(chunks),
        )
        return MediaIngestionResult(document=document, content=excerpt, meta=document.meta)

    async def _transcribe(self, data: FileData) -> str | None:
        if self.stt is None:
            return None
        return (await self.stt.transcribe(data)).strip()

    async def _analyze_image(self, data: FileData) -> str | None:
        if self.vision is None:
            return None
        return (await self.vision.describe(data)).strip()

    async def _parse_text(self, data: FileData) -> str | None:
        try:
            return (await self.registry.parse(data)).strip()
        except UnsupportedMediaError as exc:
            logger.warning("media_kind_unsupported", kind=data.kind.value, error=str(exc))
            return None
        except ParseError as exc:
            logger.warning("media_parse_failed", kind=data.kind.value, error=str(exc))
            return None


__all__ = [
    "FileFetcher",
    "IngestionPipeline",
    "SpeechToText",
    "VisionAnalyzer",
]
