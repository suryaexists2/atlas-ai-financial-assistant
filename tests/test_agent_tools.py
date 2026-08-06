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
