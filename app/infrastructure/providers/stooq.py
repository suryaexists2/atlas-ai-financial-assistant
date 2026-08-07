"""Market-index quotes via stooq's free public CSV endpoint.

No API key required. Tracks the three headline US indices so the agent can
answer "how are the markets doing" without guessing.
"""

from __future__ import annotations

import csv
import io
from typing import Any

import httpx

_INDEXES = {
    "^spx": "S&P 500",
    "^dji": "Dow Jones",
    "^ndq": "Nasdaq Composite",
}

_URL = "https://stooq.com/q/l/?s={symbols}&f=sd2t2ohlcv&h&e=csv"


class MarketIndexError(RuntimeError):
    pass


class MarketIndicesClient:
    def __init__(
        self,
        *,
        timeout_seconds: float = 15.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._http = http or httpx.AsyncClient(timeout=timeout_seconds)

    async def fetch(self) -> list[dict[str, Any]]:
        symbols = ",".join(_INDEXES.keys())
        try:
            response = await self._http.get(_URL.format(symbols=symbols))
        except httpx.HTTPError as exc:
            raise MarketIndexError(f"index service is unreachable: {exc}") from exc
        if response.status_code != 200:
            raise MarketIndexError("market index feed returned a non-200 status")
        rows = list(csv.DictReader(io.StringIO(response.text)))
        result: list[dict[str, Any]] = []
        for row in rows:
            symbol = (row.get("Symbol") or "").lower()
            if symbol not in _INDEXES:
                continue
            price = _float(row.get("Close"))
            change = _float(row.get("Change"))
            change_pct = _float(row.get("% Change"))
            if price is None:
                continue
            result.append(
                {
                    "symbol": _INDEXES[symbol],
                    "code": symbol[1:].upper(),
                    "price": price,
                    "change": change,
                    "change_percent": change_pct,
                }
            )
        return result


def _float(value: str | None) -> float | None:
    if value in (None, "", "N/D"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


__all__ = ["MarketIndicesClient", "MarketIndexError"]
