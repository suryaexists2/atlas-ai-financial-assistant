"""Google Sheets access via public CSV export (no OAuth / API key needed).

Reads a Google Sheet that is shared with "Anyone with the link" (or published
to the web) using the escaped-CSV endpoint. This keeps the integration
zero-config: a user pastes a spreadsheet URL and Atlas can query its rows.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any

import httpx

_SHEET_ID_RE = re.compile(r"/spreadsheets/d/([a-zA-Z0-9_-]{10,})")

_UA = "Mozilla/5.0 (compatible; AtlasAI/0.1; +https://atlas-bot-peop.onrender.com)"


class GoogleSheetsError(RuntimeError):
    pass


class GoogleSheetsClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._http = http or httpx.AsyncClient(timeout=timeout_seconds)

    async def fetch_rows(self, url: str, *, max_rows: int = 200) -> list[dict[str, Any]]:
        """Returns up to `max_rows` rows as dicts keyed by the header row."""
        sheet_id = sheet_id_from_url(url)
        if sheet_id is None:
            raise GoogleSheetsError(
                "I couldn't find a Google Sheet id in that link. Send the "
                "standard /spreadsheets/d/<id>/edit share link."
            )

        target = (
            f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
        )
        try:
            response = await self._http.get(target, headers={"User-Agent": _UA})
        except httpx.HTTPError as exc:
            raise GoogleSheetsError(f"Google Sheets is unreachable: {exc}") from exc
        if response.status_code != 200:
            raise GoogleSheetsError(
                "Google Sheets returned an error. Make sure the sheet is shared "
                "with 'Anyone with the link' (viewer), then resend the link."
            )
        rows = _parse_csv(response.text)
        if not rows:
            raise GoogleSheetsError("That sheet is empty or could not be read as CSV.")
        return rows[:max_rows]

    async def aclose(self) -> None:
        await self._http.aclose()


def sheet_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = _SHEET_ID_RE.search(url)
    return match.group(1) if match else None


def _parse_csv(text: str) -> list[dict[str, Any]]:
    """Parses the CSV body (header row + quoted fields) into dicts."""
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


__all__ = ["GoogleSheetsClient", "GoogleSheetsError", "sheet_id_from_url"]
