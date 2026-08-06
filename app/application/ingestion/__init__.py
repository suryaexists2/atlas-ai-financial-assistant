"""Media/document ingestion: types, ports, pipeline.

The pipeline turns a raw Telegram attachment (voice note, image, PDF, CSV,
spreadsheet, …) into plain text the agent can reason over. Parsers are small
adapters that own one format family; they are injected so tests use fakes
and infrastructure stays swappable.
"""

from app.application.ingestion.chunker import chunk_text
from app.application.ingestion.pipeline import IngestionPipeline
from app.application.ingestion.types import (
    FileData,
    MediaIngestionResult,
    ParsedDocument,
    ParseError,
    UnsupportedMediaError,
)
from app.domain.enums import DocumentKind

__all__ = [
    "DocumentKind",
    "FileData",
    "IngestionPipeline",
    "MediaIngestionResult",
    "ParsedDocument",
    "ParseError",
    "UnsupportedMediaError",
    "chunk_text",
]
