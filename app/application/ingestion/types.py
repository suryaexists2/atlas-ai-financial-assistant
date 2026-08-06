"""Core ingestion data shapes and errors (application-layer domain)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.domain.enums import DocumentKind


class ParseError(RuntimeError):
    """A parser was selected for the payload but could not extract text."""


class UnsupportedMediaError(RuntimeError):
    """No parser exists for this payload kind (or a service is missing)."""


@dataclass(frozen=True)
class FileData:
    """Raw attachment bytes after download, before any parsing."""

    raw: bytes
    filename: str | None = None
    mime_type: str | None = None
    size: int = 0

    @property
    def kind(self) -> DocumentKind:
        return detect_kind(self.mime_type, self.filename)


@dataclass
class ParsedDocument:
    """Result of parsing one attachment into text."""

    kind: DocumentKind
    text: str
    filename: str | None = None
    mime_type: str | None = None
    byte_size: int = 0
    chunk_count: int = 1
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MediaIngestionResult:
    """Outcome of one ingestion attempt for the processor/reply layer."""

    document: ParsedDocument | None = None
    content: str | None = None  # what the model should see as the user turn
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None  # human-readable failure (never leaked raw)
    error_code: str | None = None  # machine-readable stage ("download", "parse", "stt", ...)

    @property
    def ok(self) -> bool:
        return self.error is None and self.document is not None


_MIME_EXT_KIND: tuple[tuple[tuple[str, ...], tuple[str, ...], DocumentKind], ...] = (
    (("text/plain",), (".txt",), DocumentKind.TEXT),
    (("text/markdown", "text/x-markdown"), (".md", ".markdown"), DocumentKind.MARKDOWN),
    (("text/csv",), (".csv",), DocumentKind.CSV),
    (("application/json",), (".json",), DocumentKind.JSON),
    (("application/pdf",), (".pdf",), DocumentKind.PDF),
    (
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",),
        (".docx",),
        DocumentKind.DOCX,
    ),
    (
        (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        ),
        (".xlsx", ".xls"),
        DocumentKind.XLSX,
    ),
    (
        (
            "audio/ogg",
            "audio/mpeg",
            "audio/mp3",
            "audio/wav",
            "audio/x-m4a",
            "audio/webm",
            "audio/opus",
            "audio/mp4",
        ),
        (".ogg", ".oga", ".mp3", ".wav", ".m4a", ".webm", ".opus"),
        DocumentKind.VOICE,
    ),
    (
        ("image/jpeg", "image/png", "image/gif", "image/webp", "image/heic", "image/bmp"),
        (".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".bmp"),
        DocumentKind.IMAGE,
    ),
)


def detect_kind(mime_type: str | None, filename: str | None) -> DocumentKind:
    """Classifies a payload by mime type first, then by file extension."""
    mime = (mime_type or "").lower().split(";")[0].strip()
    for mimes, _, kind in _MIME_EXT_KIND:
        if mime and mime in mimes:
            return kind
    name = (filename or "").lower()
    for _, exts, kind in _MIME_EXT_KIND:
        if name.endswith(exts):
            return kind
    return DocumentKind.UNSUPPORTED
