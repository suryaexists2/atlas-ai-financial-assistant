"""Google Calendar API client: list upcoming events and create meetings."""

from __future__ import annotations

import datetime as dt
from typing import Any

import httpx

from app.infrastructure.providers.google_oauth import (
    GoogleApiError,
    GoogleTokenExpiredError,
)

_API_BASE = "https://www.googleapis.com/calendar/v3/calendars/primary"
_UA = "Mozilla/5.0 (compatible; AtlasAI/0.1; +https://atlas-bot-peop.onrender.com)"


class CalendarClient:
    def __init__(
        self,
        access_token: str,
        *,
        timeout_seconds: float = 30.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._token = access_token
        self._http = http or httpx.AsyncClient(timeout=timeout_seconds)

    async def list_events(
        self,
        *,
        days: int = 7,
        max_results: int = 30,
    ) -> list[dict[str, Any]]:
        now = dt.datetime.now(dt.UTC)
        time_min = now.isoformat()
        time_max = (now + dt.timedelta(days=days)).isoformat()
        params = {
            "timeMin": time_min,
            "timeMax": time_max,
            "orderBy": "startTime",
            "singleEvents": "true",
            "maxResults": max_results,
        }
        payload = await self._get("/events", params=params)
        return [
            {
                "id": item.get("id"),
                "summary": item.get("summary"),
                "start": (item.get("start") or {}).get("dateTime")
                or (item.get("start") or {}).get("date"),
                "end": (item.get("end") or {}).get("dateTime")
                or (item.get("end") or {}).get("date"),
                "location": item.get("location"),
                "description": (item.get("description") or "")[:500],
            }
            for item in payload.get("items", [])
        ]

    async def create_event(
        self,
        *,
        summary: str,
        start: dt.datetime,
        end: dt.datetime,
        description: str | None = None,
        attendees: list[str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "summary": summary,
            "start": {"dateTime": start.isoformat()},
            "end": {"dateTime": end.isoformat()},
        }
        if description:
            body["description"] = description
        if attendees:
            body["attendees"] = [{"email": email} for email in attendees]
        return await self._post("/events", body=body)

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
            raise GoogleApiError("Calendar API is unreachable") from exc
        if response.status_code in (401, 403):
            raise GoogleTokenExpiredError()
        if response.status_code != 200:
            raise GoogleApiError(
                f"Calendar API error ({response.status_code}): {response.text[:200]}",
                status=response.status_code,
            )
        return response.json()

    async def _post(self, path: str, *, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._http.post(
                f"{_API_BASE}{path}",
                json=body,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "User-Agent": _UA,
                    "Content-Type": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            raise GoogleApiError("Calendar API is unreachable") from exc
        if response.status_code in (401, 403):
            raise GoogleTokenExpiredError()
        if response.status_code >= 300:
            raise GoogleApiError(
                f"Calendar API error ({response.status_code}): {response.text[:200]}",
                status=response.status_code,
            )
        return response.json()


__all__ = ["CalendarClient"]
