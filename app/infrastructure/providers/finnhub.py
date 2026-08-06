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
        data = response.json()
        if not isinstance(data, dict):
            raise FinnhubError("Finnhub returned an unexpected response shape")
        return data


__all__ = ["FinnhubClient", "FinnhubError"]
