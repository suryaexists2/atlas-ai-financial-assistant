"""Heuristic chunking for very large extracted documents."""

from __future__ import annotations


def chunk_text(text: str, *, max_chars: int, overlap: int = 200) -> list[str]:
    """Splits text into overlapping chunks of at most `max_chars`.

    Splits on paragraph breaks when possible (keeps financial tables intact);
    preserves a small overlap so a sentence straddling two chunks is still
    readable if chunks are ever re-read individually.
    """
    if max_chars <= 0:
        max_chars = 12_000
    if len(text) <= max_chars:
        return [text] if text else []

    blocks = _split_blocks(text) if max_chars >= 4_000 else [text]
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(block, max_chars=max_chars, overlap=overlap))
            continue
        if current and len(current) + 1 + len(block) > max_chars:
            chunks.append(current)
            current = f"{current[-overlap:]}\n{block}"
        else:
            current = f"{current}\n{block}" if current else block
    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


def _split_blocks(text: str) -> list[str]:
    parts = text.replace("\r\n", "\n").split("\n")
    blocks: list[str] = []
    current: list[str] = []
    for line in parts:
        if line.strip() == "":
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks or [text]


def _hard_split(text: str, *, max_chars: int, overlap: int = 200) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(start, end - overlap)
    return chunks


def truncate_excerpt(text: str, *, max_chars: int) -> tuple[str, bool]:
    """Returns (bounded excerpt, was_truncated)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


__all__ = ["chunk_text", "truncate_excerpt"]
