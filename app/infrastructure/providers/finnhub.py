"""Finnhub REST client (free tier).

Provides quote and company-profile lookups used by the agent's market tools.
"""

from __future__ import annotations

from typing import Any

import httpx


class FinnhubError(RuntimeError):
    pass


class FinnhubClient:
    BASE = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 15.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._http = http or httpx.AsyncClient(timeout=timeout_seconds)

    async def quote(self, symbol: str) -> dict[str, Any]:
        """Current quote: c=current, d=change, dp=change %, pc=prev close."""
        return await self._get("/quote", {"symbol": symbol})

    async def company_profile(self, symbol: str) -> dict[str, Any]:
        """Company profile 2: name, exchange, industry, market cap, etc."""
        return await self._get("/stock/profile2", {"symbol": symbol})

    async def general_news(self, *, limit: int = 10) -> list[dict[str, Any]]:
        """Latest market news headlines (general category)."""
        data = await self._get("/news", {"category": "general"})
        if not isinstance(data, list):
            return []
        return [self._news_item(item) for item in data[:limit]]

    async def company_news(
        self,
        symbol: str,
        *,
        from_date: str | None = None,
        to_date: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Recent company-specific news for a ticker."""
        params = {"symbol": symbol}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        data = await self._get("/company-news", params)
        if not isinstance(data, list):
            return []
        return [self._news_item(item) for item in data[:limit]]

    async def earnings(self, symbol: str) -> dict[str, Any]:
        """Upcoming/latest earnings dates for a ticker (single next event)."""
        data = await self._get("/stock/earnings", {"symbol": symbol, "limit": 3})
        if not isinstance(data, list) or not data:
            return {}
        return data[0]

    @staticmethod
    def _news_item(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "headline": item.get("headline"),
            "source": item.get("source"),
            "url": item.get("url"),
            "datetime": item.get("datetime"),
            "summary": (item.get("summary") or "")[:240],
            "related": item.get("related"),
        }

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._http.get(
                f"{self.BASE}{path}",
                params={**params, "token": self._api_key},
            )
        except httpx.HTTPError as exc:
            raise FinnhubError(f"Finnhub request failed: {exc}") from exc
        if response.status_code >= 400:
            raise FinnhubError(f"Finnhub error {response.status_code}: {response.text[:200]}")
        return response.json()


__all__ = ["FinnhubClient", "FinnhubError"]
