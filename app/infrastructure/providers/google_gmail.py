"""Gmail API client (read-only): search and fetch emails for summarization.

Only headers (From/Subject/Date) plus a bounded plain-text body excerpt are
returned — full message content is never logged or persisted beyond what the
LLM tool result carries.
"""

from __future__ import annotations

import base64
import html as _html
import re
from typing import Any

import httpx

from app.infrastructure.providers.google_oauth import (
    GoogleApiError,
    GoogleTokenExpiredError,
)

_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
_MAX_BODY_CHARS = 3000
_UA = "Mozilla/5.0 (compatible; AtlasAI/0.1; +https://atlas-bot-peop.onrender.com)"

_TAG_RE = re.compile(r"<[^>]+>")


class GmailClient:
    def __init__(
        self,
        access_token: str,
        *,
        timeout_seconds: float = 30.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._token = access_token
        self._http = http or httpx.AsyncClient(timeout=timeout_seconds)

    async def search(self, query: str, *, max_results: int = 20) -> list[dict[str, Any]]:
        params = {"q": query, "maxResults": max_results}
        payload = await self._get("/messages", params=params)
        return [
            {
                "id": item.get("id"),
                "thread_id": item.get("threadId"),
                "snippet": item.get("snippet"),
            }
            for item in payload.get("messages", [])
        ]

    async def get_message(
        self, message_id: str, *, max_body_chars: int = _MAX_BODY_CHARS
    ) -> dict[str, Any]:
        payload = await self._get(
            f"/messages/{message_id}",
            params={"format": "full", "metadataHeaders": "From,Subject,Date"},
        )
        headers: dict[str, str] = {}
        for part in payload.get("payload", {}).get("headers", []):
            name = part.get("name", "").lower()
            if name in ("from", "subject", "date"):
                headers[name] = part.get("value", "")
        body = _extract_body(payload.get("payload", {}))[:max_body_chars]
        return {
            "id": message_id,
            "from": headers.get("from"),
            "subject": headers.get("subject"),
            "date": headers.get("date"),
            "body_excerpt": body,
        }

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _get(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._http.get(
                f"{_API_BASE}{path}",
                params=params,
                headers={"Authorization": f"Bearer {self._token}", "User-Agent": _UA},
            )
        except httpx.HTTPError as exc:
            raise GoogleApiError("Gmail API is unreachable") from exc
        if response.status_code in (401, 403):
            raise GoogleTokenExpiredError()
        if response.status_code != 200:
            raise GoogleApiError(
                f"Gmail API error ({response.status_code})", status=response.status_code
            )
        return response.json()


def _extract_body(payload: dict[str, Any]) -> str:
    """Walks the MIME payload for plain text (fallback: stripped HTML/text)."""
    mime = payload.get("mimeType", "")
    data = payload.get("body", {}).get("data")
    if data:
        try:
            raw = base64.urlsafe_b64decode(data.encode("ascii")).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - malformed body is not fatal
            raw = ""
        if mime == "text/plain":
            return raw
        if mime == "text/html":
            return _TAG_RE.sub(" ", _html.unescape(raw))
    for part in payload.get("parts", []):
        text = _extract_body(part)
        if text:
            return text
    return ""


__all__ = ["GmailClient"]
