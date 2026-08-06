"""Finnhub + SEC EDGAR provider client tests (against MockTransport)."""

import httpx
import pytest

from app.infrastructure.providers.finnhub import FinnhubClient, FinnhubError
from app.infrastructure.providers.sec import SecEdgarClient, SecError

# --- Finnhub ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_finnhub_quote():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["symbol"] == "AAPL"
        return httpx.Response(
            200,
            json={
                "c": 250.1,
                "d": 2.3,
                "dp": 0.93,
                "h": 252.0,
                "l": 248.0,
                "o": 249.5,
                "pc": 247.8,
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = FinnhubClient("key", http=http)
    quote = await client.quote("AAPL")
    assert quote["c"] == 250.1
    assert quote["dp"] == 0.93


@pytest.mark.asyncio
async def test_finnhub_error_propagates():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = FinnhubClient("bad-key", http=http)
    with pytest.raises(FinnhubError):
        await client.quote("AAPL")


# --- SEC EDGAR ---------------------------------------------------------


def _tickers_payload():
    return {"0": {"ticker": "AAPL", "cik_str": 320193}, "1": {"ticker": "MSFT", "cik_str": 789019}}


def _submissions_payload():
    return {
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "8-K", "10-Q"],
                "filingDate": ["2026-01-15", "2025-11-05", "2025-10-20", "2025-08-01"],
                "reportDate": ["2025-09-27", "2025-07-04", "2025-10-15", "2025-04-09"],
                "accessionNumber": [
                    "0001",
                    "0002",
                    "0003",
                    "0004",
                ],
                "primaryDocument": ["a1.htm", "a2.htm", "a3.htm", "a4.htm"],
            }
        }
    }


@pytest.mark.asyncio
async def test_sec_resolves_ticker_and_returns_filings():
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "company_tickers" in url:
            return httpx.Response(200, json=_tickers_payload())
        assert "submissions/CIK0000320193" in url
        return httpx.Response(200, json=_submissions_payload())

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = SecEdgarClient("test contact@example.com", http=http)
    assert await client.cik_for_ticker("aapl") == 320193

    filings = await client.recent_filings("AAPL", limit=3)
    assert [f["form"] for f in filings] == ["10-K", "10-Q", "8-K"]
    assert filings[0]["filed_on"] == "2026-01-15"
    assert filings[0]["url"].startswith("https://www.sec.gov/Archives/edgar/data/0000320193")


@pytest.mark.asyncio
async def test_sec_returns_empty_for_unknown_ticker():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_tickers_payload())

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = SecEdgarClient("http contact@example.com", http=http)
    assert await client.recent_filings("ZZZZ", limit=3) == []


@pytest.mark.asyncio
async def test_sec_error_propagates():
    def handler(request: httpx.Request) -> httpx.Response:
        if "company_tickers" in str(request.url):
            return httpx.Response(200, json=_tickers_payload())
        return httpx.Response(500, text="boom")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = SecEdgarClient("http contact@example.com", http=http)
    with pytest.raises(SecError):
        await client.recent_filings("AAPL", limit=3)
