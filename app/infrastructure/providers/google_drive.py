"""Google Drive API client: search files and download bytes for summarization.

Native Google formats (Docs/Sheets/Slides) are exported as plain text via the
export endpoint; binary formats (PDF/DOCX/XLSX/TXT/MD/CSV/JSON) are downloaded
as bytes so the existing ingestion pipeline can parse them.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.infrastructure.providers.google_oauth import (
    GoogleApiError,
    GoogleTokenExpiredError,
)

_API_BASE = "https://www.googleapis.com/drive/v3"
_UA = "Mozilla/5.0 (compatible; AtlasAI/0.1; +https://atlas-bot-peop.onrender.com)"

_GOOGLE_MIME_TO_TEXT = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
    "application/vnd.google-apps.drawing": "image/png",
}
_MAX_BYTES = 25_000_000


class DriveClient:
    def __init__(
        self,
        access_token: str,
        *,
        timeout_seconds: float = 60.0,
        max_bytes: int = _MAX_BYTES,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._token = access_token
        self._max_bytes = max_bytes
        self._http = http or httpx.AsyncClient(timeout=timeout_seconds)

    async def search(self, query: str, *, max_results: int = 20) -> list[dict[str, Any]]:
        params = {
            "q": query,
            "fields": "files(id,name,mimeType,size,modifiedTime)",
            "pageSize": max_results,
        }
        payload = await self._get("/files", params=params)
        return [
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "mime_type": item.get("mimeType"),
                "size": item.get("size"),
                "modified_time": item.get("modifiedTime"),
            }
            for item in payload.get("files", [])
        ]

    async def download(
        self, file_id: str, *, mime_type: str | None = None, filename: str = "drive_file"
    ) -> bytes:
        """Downloads a file. Google-native formats are exported to text/CSV."""
        if mime_type in _GOOGLE_MIME_TO_TEXT:
            export_mime = _GOOGLE_MIME_TO_TEXT[mime_type]
            url = f"{_API_BASE}/files/{file_id}/export"
            params = {"mimeType": export_mime}
        else:
            url = f"{_API_BASE}/files/{file_id}"
            params = {"alt": "media"}
        try:
            response = await self._http.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {self._token}", "User-Agent": _UA},
            )
        except httpx.HTTPError as exc:
            raise GoogleApiError("Drive API is unreachable") from exc
        if response.status_code in (401, 403):
            raise GoogleTokenExpiredError()
        if response.status_code != 200:
            raise GoogleApiError(
                f"Drive API error ({response.status_code})", status=response.status_code
            )
        if len(response.content) > self._max_bytes:
            raise GoogleApiError(f"Drive file too large to analyze ({len(response.content)} bytes)")
        return response.content

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
            raise GoogleApiError("Drive API is unreachable") from exc
        if response.status_code in (401, 403):
            raise GoogleTokenExpiredError()
        if response.status_code != 200:
            raise GoogleApiError(
                f"Drive API error ({response.status_code})", status=response.status_code
            )
        return response.json()


__all__ = ["DriveClient"]
