"""Parser port and registry.

A `ContentParser` turns a `FileData` into plain-text. The registry picks the
parser for a detected `DocumentKind`. Parsers are constructed out-of-tree (in
`app/infrastructure/ingestion/parsers.py`) and injected so the application
layer never depends on aiogram, pypdf, openpyxl, or the LLM clients.
"""

from __future__ import annotations

from typing import Protocol

from app.application.ingestion.types import FileData, ParseError, UnsupportedMediaError
from app.domain.enums import DocumentKind


class ContentParser(Protocol):
    """Extracts plain text from one format family."""

    kind: DocumentKind

    async def parse(self, data: FileData) -> str:
        """Returns extracted text; raises ParseError on extraction failure."""
        ...


class ParserRegistry:
    def __init__(self, parsers: list[ContentParser]) -> None:
        by_kind: dict[DocumentKind, ContentParser] = {}
        for parser in parsers:
            by_kind[parser.kind] = parser
        self._parsers = by_kind

    def get(self, kind: DocumentKind) -> ContentParser | None:
        return self._parsers.get(kind)

    async def parse(self, data: FileData) -> str:
        """Parses with the parser for `data.kind`; raises UnsupportedMediaError
        when none is registered and ParseError on adapter failure."""
        parser = self._parsers.get(data.kind)
        if parser is None:
            raise UnsupportedMediaError(f"no parser registered for kind={data.kind}")
        try:
            return await parser.parse(data)
        except (UnsupportedMediaError, ParseError):
            raise
        except Exception as exc:  # noqa: BLE001 - a broken adapter must not crash ingestion
            raise ParseError(f"{data.kind} parsing failed: {exc}") from exc

    def __len__(self) -> int:
        return len(self._parsers)


__all__ = ["ContentParser", "ParserRegistry"]
