"""Infrastructure adapters for media/document ingestion.

Downloading lives here (needs the Telegram Bot), the AI-based parsers (STT,
vision) live here (need provider credentials), and the pure-file parsers live
here too so the whole feature stays behind the application-layer ports.
"""

from app.infrastructure.ingestion.downloader import TelegramFileFetcher, TooLargeError
from app.infrastructure.ingestion.media_ai import OpenRouterMediaAI
from app.infrastructure.ingestion.parsers import (
    CsvParser,
    DocxParser,
    ImageParser,
    JsonParser,
    MarkdownParser,
    PdfParser,
    TextParser,
    VoiceParser,
    XlsxParser,
    build_default_registry,
)

__all__ = [
    "CsvParser",
    "DocxParser",
    "ImageParser",
    "JsonParser",
    "MarkdownParser",
    "OpenRouterMediaAI",
    "PdfParser",
    "TelegramFileFetcher",
    "TextParser",
    "TooLargeError",
    "VoiceParser",
    "XlsxParser",
    "build_default_registry",
]
