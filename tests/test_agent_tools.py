"""Agent tool registry tests: schemas, execution, provider and DB integration."""

import json

import pytest

from app.application.agent.tools import ToolContext, default_registry


@pytest.fixture
def registry():
    return default_registry()


def test_schemas_are_openai_shaped(registry):
    schemas = registry.schemas()
    names = {s["function"]["name"] for s in schemas}
    assert "get_market_quote" in names
    assert "save_memory" in names
    assert "list_watchlist" in names
    for schema in schemas:
        assert schema["type"] == "function"
        assert "parameters" in schema["function"]


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_json(registry, uow, demo_user):
    ctx = ToolContext(uow=uow, user_id=demo_user["user_id"])
    result = await registry.execute(ctx, "no_such_tool", {})
    assert json.loads(result)["error"].startswith("unknown tool")


@pytest.mark.asyncio
async def test_quote_tool_uses_provider(uow, demo_user):
    import httpx

    from app.infrastructure.providers.finnhub import FinnhubClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"c": 100.0, "d": 1.0, "dp": 1.01, "h": 101.0, "l": 99.0, "o": 99.5, "pc": 99.0},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    finnhub = FinnhubClient("k", http=http)
    registry = default_registry()
    ctx = ToolContext(uow=uow, user_id=demo_user["user_id"], finnhub=finnhub)
    result = json.loads(await registry.execute(ctx, "get_market_quote", {"symbol": "aapl"}))
    assert result["symbol"] == "AAPL"
    assert result["current"] == 100.0


@pytest.mark.asyncio
async def test_filings_tool_uses_provider(uow, demo_user):
    import httpx

    from app.infrastructure.providers.sec import SecEdgarClient

    def handler(request: httpx.Request) -> httpx.Response:
        if "company_tickers" in str(request.url):
            return httpx.Response(200, json={"0": {"ticker": "AAPL", "cik_str": 320193}})
        return httpx.Response(
            200,
            json={
                "filings": {
                    "recent": {
                        "form": ["10-K"],
                        "filingDate": ["2026-01-15"],
                        "reportDate": ["2025-09-27"],
                        "accessionNumber": ["0001"],
                        "primaryDocument": ["a.htm"],
                    }
                }
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sec = SecEdgarClient("test contact@example.com", http=http)
    registry = default_registry()
    ctx = ToolContext(uow=uow, user_id=demo_user["user_id"], sec=sec)
    result = json.loads(
        await registry.execute(ctx, "get_company_filings", {"symbol": "AAPL", "limit": 1})
    )
    assert result["filings"][0]["form"] == "10-K"


@pytest.mark.asyncio
async def test_watchlist_add_list_remove(uow, demo_user):
    registry = default_registry()
    user_id = demo_user["user_id"]

    async with uow:
        ctx = ToolContext(uow=uow, user_id=user_id)
        out = json.loads(await registry.execute(ctx, "add_to_watchlist", {"symbol": "tsla"}))
        assert "added TSLA" in out["message"]

    async with uow:
        ctx = ToolContext(uow=uow, user_id=user_id)
        listed = json.loads(await registry.execute(ctx, "list_watchlist", {}))
        assert [item["symbol"] for item in listed] == ["TSLA"]

    async with uow:
        ctx = ToolContext(uow=uow, user_id=user_id)
        removed = json.loads(
            await registry.execute(ctx, "remove_from_watchlist", {"symbol": "tsla"})
        )
        assert "removed TSLA" in removed["message"]

    async with uow:
        ctx = ToolContext(uow=uow, user_id=user_id)
        listed = json.loads(await registry.execute(ctx, "list_watchlist", {}))
        assert listed == []


@pytest.mark.asyncio
async def test_memory_save_and_list(uow, demo_user):
    registry = default_registry()
    user_id = demo_user["user_id"]

    async with uow:
        ctx = ToolContext(uow=uow, user_id=user_id)
        saved = json.loads(
            await registry.execute(
                ctx,
                "save_memory",
                {
                    "memory_key": "interest:fintech",
                    "summary": "Loves fintech stocks",
                    "confidence": 0.9,
                },
            )
        )
        assert saved["memory_key"] == "interest:fintech"

    async with uow:
        ctx = ToolContext(uow=uow, user_id=user_id)
        listed = json.loads(await registry.execute(ctx, "list_memories", {}))
        assert any(m["key"] == "interest:fintech" for m in listed)


@pytest.mark.asyncio
async def test_provider_error_is_caught_in_tool(uow, demo_user):
    import httpx

    from app.infrastructure.providers.finnhub import FinnhubClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    finnhub = FinnhubClient("k", http=http)
    registry = default_registry()
    ctx = ToolContext(uow=uow, user_id=demo_user["user_id"], finnhub=finnhub)
    result = json.loads(await registry.execute(ctx, "get_market_quote", {"symbol": "aapl"}))
    assert "error" in result


@pytest.mark.asyncio
async def test_alert_create_list_delete(uow, demo_user):
    registry = default_registry()
    user_id = demo_user["user_id"]

    async with uow:
        ctx = ToolContext(uow=uow, user_id=user_id)
        created = json.loads(
            await registry.execute(ctx, "create_price_alert", {"symbol": "aapl", "percent": 4})
        )
        assert "price alert for AAPL" in created["message"]

        listed = json.loads(await registry.execute(ctx, "list_alerts", {}))
        assert len(listed) == 1 and listed[0]["kind"] == "price"
        alert_id = listed[0]["id"]

    async with uow:
        ctx = ToolContext(uow=uow, user_id=user_id)
        deleted = json.loads(
            await registry.execute(ctx, "delete_alert", {"alert_id": str(alert_id)})
        )
        assert "removed" in deleted["message"]

    async with uow:
        ctx = ToolContext(uow=uow, user_id=user_id)
        listed = json.loads(await registry.execute(ctx, "list_alerts", {}))
        assert listed == []


@pytest.mark.asyncio
async def test_reminder_and_briefing_create_jobs(uow, demo_user):
    registry = default_registry()
    user_id = demo_user["user_id"]

    async with uow:
        ctx = ToolContext(uow=uow, user_id=user_id)
        out = json.loads(
            await registry.execute(
                ctx, "create_reminder", {"text": "prep for earnings call", "time": "7:30am"}
            )
        )
        assert "reminder set for 07:30" in out["message"]
        jobs = {j.job_type: j for j in await uow.jobs.list_enabled()}
        assert jobs["reminder"].params["text"] == "prep for earnings call"

    async with uow:
        ctx = ToolContext(uow=uow, user_id=user_id)
        out = json.loads(await registry.execute(ctx, "create_daily_briefing", {"time": "09:15"}))
        assert "scheduled at 09:15" in out["message"]
        jobs = {j.job_type: j for j in await uow.jobs.list_enabled()}
        assert jobs["daily_brief"].cron_expr == "15 9 * * *"

    async with uow:
        ctx = ToolContext(uow=uow, user_id=user_id)
        dup = json.loads(await registry.execute(ctx, "create_daily_briefing", {"time": "10:00"}))
        assert "already scheduled" in dup["message"]


@pytest.mark.asyncio
async def test_market_news_and_earnings_tools(uow, demo_user):
    import httpx

    from app.infrastructure.providers.finnhub import FinnhubClient

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "news" in url:
            return httpx.Response(
                200,
                json=[
                    {"headline": "Markets rally", "source": "Bloomberg", "url": "https://x"}
                ],
            )
        if "earnings" in url:
            return httpx.Response(
                200,
                json=[{"symbol": "NVDA", "date": "2026-08-20", "quarter": 2}],
            )
        return httpx.Response(500, text="unexpected")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    finnhub = FinnhubClient("k", http=http)
    registry = default_registry()

    async with uow:
        ctx = ToolContext(uow=uow, user_id=demo_user["user_id"], finnhub=finnhub)
        news = json.loads(await registry.execute(ctx, "get_market_news", {"limit": 5}))
        assert news[0]["headline"] == "Markets rally"

        company = json.loads(
            await registry.execute(ctx, "get_company_news", {"symbol": "aapl", "limit": 3})
        )
        assert company["symbol"] == "AAPL"

        earnings = json.loads(
            await registry.execute(ctx, "get_company_earnings", {"symbol": "nvda"})
        )
        assert earnings["earnings"]["date"] == "2026-08-20"


@pytest.mark.asyncio
async def test_get_document_contents_returns_extracted_text(uow, demo_user):
    registry = default_registry()
    user_id = demo_user["user_id"]

    async with uow:
        await uow.documents.create(
            user_id,
            filename="report.pdf",
            doc_meta={"extracted_text": "Revenue grew 20% in Q2", "kind": "PDF"},
        )
        await uow.commit()

    async with uow:
        ctx = ToolContext(uow=uow, user_id=user_id)
        contents = json.loads(await registry.execute(ctx, "get_document_contents", {"index": 0}))
        assert contents["filename"] == "report.pdf"
        assert "Revenue grew 20%" in contents["text"]


@pytest.mark.asyncio
async def test_market_indices_tool(uow, demo_user):
    import httpx

    from app.infrastructure.providers.stooq import MarketIndicesClient

    CSV = (
        "Symbol,Date,Time,Open,High,Low,Close,Volume,Change,% Change\n"
        "^spx,2026-08-07,,,,,5900.00,,,+0.40\n"
        "^dji,2026-08-07,,,,,41000.00,,,-0.10\n"
        "^ndq,2026-08-07,,,,,19900.00,,,+1.20\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=CSV)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    indices = MarketIndicesClient(http=http)
    registry = default_registry()

    async with uow:
        ctx = ToolContext(uow=uow, user_id=demo_user["user_id"], indices=indices)
        out = json.loads(await registry.execute(ctx, "get_market_indices", {}))
        assert out["count"] == 3
        codes = {i["code"] for i in out["indices"]}
        assert codes == {"SPX", "DJI", "NDQ"}
        assert out["indices"][0]["price"] == 5900.0


@pytest.mark.asyncio
async def test_link_and_read_google_sheet(uow, demo_user):
    import httpx

    from app.infrastructure.providers.google_sheets import GoogleSheetsClient

    SHEET_URL = "https://docs.google.com/spreadsheets/d/abc123XYZuvw45/edit"
    CSV = 'ticker,qty\nAAPL,100\nTSLA,50\n'

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/abc123XYZuvw45/gviz/tq" in str(request.url)
        return httpx.Response(200, text=CSV)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sheets = GoogleSheetsClient(http=http)
    registry = default_registry()

    async with uow:
        ctx = ToolContext(uow=uow, user_id=demo_user["user_id"], google_sheets=sheets)
        linked = json.loads(
            await registry.execute(ctx, "link_google_sheet", {"url": SHEET_URL})
        )
        assert "linked" in linked["message"]

    async with uow:
        ctx = ToolContext(uow=uow, user_id=demo_user["user_id"], google_sheets=sheets)
        read = json.loads(await registry.execute(ctx, "read_google_sheet", {}))
        assert read["row_count"] == 2
        assert read["rows"][0]["ticker"] == "AAPL"

    async with uow:
        ctx = ToolContext(uow=uow, user_id=demo_user["user_id"], google_sheets=sheets)
        unlinked = json.loads(await registry.execute(ctx, "unlink_google_sheet", {}))
        assert "unlinked" in unlinked["message"]
