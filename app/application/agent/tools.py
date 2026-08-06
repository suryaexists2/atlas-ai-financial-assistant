"""Agent tool registry: schemas + handlers for the LLM's tool-calling.

A `Tool` pairs an OpenAI-style function schema (shown to the model) with an
async handler that receives a `ToolContext` (uow + providers + user) and
returns a plain string the model can read back. Handlers never touch the
LLM or the transport layer.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.domain.repositories import MemoryRepository, WatchlistRepository
from app.infrastructure.db.uow import UnitOfWork
from app.infrastructure.providers.finnhub import FinnhubClient, FinnhubError
from app.infrastructure.providers.sec import SecEdgarClient, SecError


@dataclass
class ToolContext:
    uow: UnitOfWork
    user_id: uuid.UUID
    finnhub: FinnhubClient | None = None
    sec: SecEdgarClient | None = None

    @property
    def watchlist(self) -> WatchlistRepository:
        return self.uow.watchlist

    @property
    def memories(self) -> MemoryRepository:
        return self.uow.memories


ToolHandler = Callable[[ToolContext, dict[str, Any]], Awaitable[str]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    required: list[str] = field(default_factory=list)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": self.parameters},
                **({"required": self.required} if self.required else {}),
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    async def execute(self, ctx: ToolContext, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps({"error": f"unknown tool: {name}"})
        try:
            result = await tool.handler(ctx, arguments)
        except (FinnhubError, SecError, ValueError) as exc:
            return json.dumps({"error": str(exc)})
        return result


# --- Handlers ----------------------------------------------------------------


async def _get_quote(ctx: ToolContext, args: dict[str, Any]) -> str:
    if ctx.finnhub is None:
        return json.dumps({"error": "market data is not configured"})
    symbol = str(args.get("symbol", "")).upper()
    if not symbol:
        return json.dumps({"error": "symbol is required"})
    quote = await ctx.finnhub.quote(symbol)
    if not quote or quote.get("c") is None:
        return json.dumps({"error": f"no quote data for {symbol}", "symbol": symbol})
    return json.dumps(
        {
            "symbol": symbol,
            "current": quote.get("c"),
            "change": quote.get("d"),
            "change_percent": quote.get("dp"),
            "high": quote.get("h"),
            "low": quote.get("l"),
            "open": quote.get("o"),
            "prev_close": quote.get("pc"),
        }
    )


async def _get_company_profile(ctx: ToolContext, args: dict[str, Any]) -> str:
    if ctx.finnhub is None:
        return json.dumps({"error": "market data is not configured"})
    symbol = str(args.get("symbol", "")).upper()
    if not symbol:
        return json.dumps({"error": "symbol is required"})
    profile = await ctx.finnhub.company_profile(symbol)
    if not profile or not profile.get("name"):
        return json.dumps({"error": f"no profile data for {symbol}", "symbol": symbol})
    return json.dumps(
        {
            "symbol": symbol,
            "name": profile.get("name"),
            "exchange": profile.get("exchange"),
            "industry": profile.get("finnhubIndustry"),
            "market_cap": profile.get("marketCapitalization"),
            "ipo": profile.get("ipo"),
            "currency": profile.get("currency"),
        }
    )


async def _get_filings(ctx: ToolContext, args: dict[str, Any]) -> str:
    if ctx.sec is None:
        return json.dumps({"error": "SEC data is not configured"})
    symbol = str(args.get("symbol", "")).upper()
    if not symbol:
        return json.dumps({"error": "symbol is required"})
    forms = args.get("form_types")
    limit = int(args.get("limit", 5))
    filings = await ctx.sec.recent_filings(symbol, form_types=forms, limit=limit)
    if not filings:
        return json.dumps({"error": f"no filings found for {symbol}", "symbol": symbol})
    return json.dumps({"symbol": symbol, "filings": filings})


async def _list_watchlist(ctx: ToolContext, args: dict[str, Any]) -> str:
    items = await ctx.watchlist.list_active(ctx.user_id)
    return json.dumps(
        [{"symbol": item.symbol, "name": item.name, "sector": item.sector} for item in items]
    )


async def _add_to_watchlist(ctx: ToolContext, args: dict[str, Any]) -> str:
    symbol = str(args.get("symbol", "")).upper()
    if not symbol:
        return json.dumps({"error": "symbol is required"})
    existing = await ctx.watchlist.get_by_symbol(ctx.user_id, symbol)
    if existing is not None:
        return json.dumps({"message": f"{symbol} is already on your watchlist"})
    await ctx.watchlist.add(
        ctx.user_id,
        symbol=symbol,
        name=args.get("name"),
        sector=args.get("sector"),
    )
    await ctx.uow.commit()
    return json.dumps({"message": f"added {symbol} to your watchlist"})


async def _remove_from_watchlist(ctx: ToolContext, args: dict[str, Any]) -> str:
    symbol = str(args.get("symbol", "")).upper()
    item = await ctx.watchlist.get_by_symbol(ctx.user_id, symbol)
    if item is None:
        return json.dumps({"message": f"{symbol} is not on your watchlist"})
    await ctx.watchlist.deactivate(item)
    await ctx.uow.commit()
    return json.dumps({"message": f"removed {symbol} from your watchlist"})


async def _save_memory(ctx: ToolContext, args: dict[str, Any]) -> str:
    key = str(args.get("memory_key", "")).strip()
    if not key:
        return json.dumps({"error": "memory_key is required"})
    memory = await ctx.memories.upsert_observation(
        ctx.user_id,
        memory_key=key,
        value=args.get("value"),
        summary=args.get("summary") or key,
        confidence=float(args.get("confidence", 0.6)),
    )
    await ctx.uow.commit()
    return json.dumps(
        {"message": "memory saved", "memory_key": key, "confidence": memory.confidence}
    )


async def _list_memories(ctx: ToolContext, args: dict[str, Any]) -> str:
    limit = int(args.get("limit", 20))
    memories = await ctx.memories.list_active(ctx.user_id, limit=limit)
    return json.dumps(
        [{"key": m.memory_key, "summary": m.summary, "confidence": m.confidence} for m in memories]
    )


DEFAULT_TOOLS: list[Tool] = [
    Tool(
        name="get_market_quote",
        description=(
            "Get the current market quote (price, change, high/low) for a US"
            " stock symbol, e.g. AAPL."
        ),
        parameters={"symbol": {"type": "string", "description": "US stock ticker symbol"}},
        required=["symbol"],
        handler=_get_quote,
    ),
    Tool(
        name="get_company_profile",
        description=(
            "Get a company profile (name, exchange, industry, market cap) for a US stock symbol."
        ),
        parameters={"symbol": {"type": "string", "description": "US stock ticker symbol"}},
        required=["symbol"],
        handler=_get_company_profile,
    ),
    Tool(
        name="get_company_filings",
        description=("Get recent SEC filings (10-K, 10-Q, 8-K by default) for a US stock symbol."),
        parameters={
            "symbol": {"type": "string", "description": "US stock ticker symbol"},
            "form_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional filing forms to include",
            },
            "limit": {"type": "integer", "description": "Max filings to return (default 5)"},
        },
        required=["symbol"],
        handler=_get_filings,
    ),
    Tool(
        name="list_watchlist",
        description="List the user's watchlist symbols.",
        parameters={},
        handler=_list_watchlist,
    ),
    Tool(
        name="add_to_watchlist",
        description="Add a stock symbol to the user's watchlist.",
        parameters={
            "symbol": {"type": "string", "description": "US stock ticker symbol"},
            "name": {"type": "string", "description": "Optional company name"},
            "sector": {"type": "string", "description": "Optional sector"},
        },
        required=["symbol"],
        handler=_add_to_watchlist,
    ),
    Tool(
        name="remove_from_watchlist",
        description="Remove a stock symbol from the user's watchlist.",
        parameters={"symbol": {"type": "string", "description": "US stock ticker symbol"}},
        required=["symbol"],
        handler=_remove_from_watchlist,
    ),
    Tool(
        name="save_memory",
        description=(
            "Remember a fact about the user (preferences, goals, risk tolerance). "
            "Use the key 'user_profile' for durable traits, 'interest:<topic>' for interests."
        ),
        parameters={
            "memory_key": {"type": "string", "description": "Stable identifier for the memory"},
            "summary": {"type": "string", "description": "Human-readable summary"},
            "value": {"type": "object", "description": "Optional structured value"},
            "confidence": {
                "type": "number",
                "description": "Confidence 0..1, default 0.6",
            },
        },
        required=["memory_key", "summary"],
        handler=_save_memory,
    ),
    Tool(
        name="list_memories",
        description="List memories stored about the user (preferences, interests, goals).",
        parameters={"limit": {"type": "integer", "description": "Max memories (default 20)"}},
        handler=_list_memories,
    ),
]


def default_registry() -> ToolRegistry:
    return ToolRegistry(DEFAULT_TOOLS)


__all__ = ["Tool", "ToolContext", "ToolRegistry", "DEFAULT_TOOLS", "default_registry"]
