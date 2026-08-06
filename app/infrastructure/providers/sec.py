"""SEC EDGAR client (keyless, rate-limited API).

Resolves tickers to CIKs via the public company tickers mapping, then reads
recent submissions (10-K, 10-Q, 8-K, ...) for a company. EDGAR requires a
User-Agent header with contact info; we use `settings.sec_user_agent`.
"""

from __future__ import annotations

from typing import Any

import httpx

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


class SecError(RuntimeError):
    pass


class SecEdgarClient:
    def __init__(
        self,
        user_agent: str,
        *,
        timeout_seconds: float = 20.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._user_agent = user_agent
        self._http = http or httpx.AsyncClient(timeout=timeout_seconds)
        self._ticker_to_cik: dict[str, int] | None = None

    async def cik_for_ticker(self, ticker: str) -> int | None:
        mapping = await self._load_tickers()
        return mapping.get(ticker.upper())

    async def recent_filings(
        self, ticker: str, *, form_types: list[str] | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Latest filings for a ticker, newest first.

        Each item: {form, filed_on, period_end, accession, url}.
        """
        cik = await self.cik_for_ticker(ticker)
        if cik is None:
            return []
        submissions = await self._get_json(_SUBMISSIONS_URL.format(cik=f"{cik:010d}"))
        recent = (submissions.get("filings") or {}).get("recent") or {}
        forms = recent.get("form") or []
        filed = recent.get("filingDate") or []
        periods = recent.get("reportDate") or []
        accessions = recent.get("accessionNumber") or []
        primary = recent.get("primaryDocument") or []

        wanted = [f.upper() for f in (form_types or ["10-K", "10-Q", "8-K"])]
        rows: list[dict[str, Any]] = []
        for i, form in enumerate(forms):
            if form.upper() not in wanted:
                continue
            accession = accessions[i] if i < len(accessions) else ""
            document = primary[i] if i < len(primary) else ""
            rows.append(
                {
                    "form": form,
                    "filed_on": filed[i] if i < len(filed) else None,
                    "period_end": periods[i] if i < len(periods) else None,
                    "accession": accession,
                    "url": (
                        _ARCHIVES_URL.format(
                            cik=f"{cik:010d}",
                            accession=accession.replace("-", ""),
                            document=document,
                        )
                        if accession and document
                        else None
                    ),
                }
            )
            if len(rows) >= limit:
                break
        return rows

    async def _load_tickers(self) -> dict[str, int]:
        if self._ticker_to_cik is not None:
            return self._ticker_to_cik
        data = await self._get_json(_TICKERS_URL)
        mapping: dict[str, int] = {}
        if isinstance(data, list):
            for entry in data:
                ticker = entry.get("ticker")
                cik = entry.get("cik_str")
                if ticker and cik:
                    mapping[ticker.upper()] = int(cik)
        elif isinstance(data, dict):
            for entry in data.values():
                ticker = entry.get("ticker")
                cik = entry.get("cik_str")
                if ticker and cik:
                    mapping[ticker.upper()] = int(cik)
        self._ticker_to_cik = mapping
        return mapping

    async def _get_json(self, url: str) -> dict[str, Any]:
        try:
            response = await self._http.get(
                url,
                headers={"User-Agent": self._user_agent},
            )
        except httpx.HTTPError as exc:
            raise SecError(f"SEC request failed: {exc}") from exc
        if response.status_code >= 400:
            if response.status_code == 404:
                return {}
            raise SecError(f"SEC error {response.status_code}: {response.text[:200]}")
        data = response.json()
        if not isinstance(data, dict):
            raise SecError("SEC returned an unexpected response shape")
        return data


__all__ = ["SecEdgarClient", "SecError"]
