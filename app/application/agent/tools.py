"""Agent tool registry: schemas + handlers for the LLM's tool-calling.

A `Tool` pairs an OpenAI-style function schema (shown to the model) with an
async handler that receives a `ToolContext` (uow + providers + user) and
returns a plain string the model can read back. Handlers never touch the
LLM or the transport layer.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.application.ingestion.types import FileData
from app.application.scheduling.cron import (
    UTC,
    compute_next_run,
    cron_from_local_time,
    extract_clock_time,
)
from app.domain.enums import AlertKind, IntegrationProvider
from app.domain.repositories import MemoryRepository, WatchlistRepository
from app.infrastructure.db.uow import UnitOfWork
from app.infrastructure.providers.finnhub import FinnhubClient, FinnhubError
from app.infrastructure.providers.google_calendar import CalendarClient
from app.infrastructure.providers.google_drive import DriveClient
from app.infrastructure.providers.google_gmail import GmailClient
from app.infrastructure.providers.google_oauth import (
    GoogleApiError,
    GoogleOAuthClient,
    GoogleOAuthError,
    GoogleTokenExpiredError,
    generate_state,
    generate_verifier,
)
from app.infrastructure.providers.sec import SecEdgarClient, SecError


@dataclass
class ToolContext:
    uow: UnitOfWork
    user_id: uuid.UUID
    finnhub: FinnhubClient | None = None
    sec: SecEdgarClient | None = None
    google_sheets: Any = None
    indices: Any = None
    google_oauth: GoogleOAuthClient | None = None
    media_pipeline: Any = None
    public_base_url: str | None = None
    chat_id: int | None = None
    # Injectable HTTP client for connector clients (tests use MockTransport).
    google_http: Any = None
    # Set by connect_google; the responder renders the inline button when present.
    oauth_connect_url: str | None = None

    @property
    def watchlist(self) -> WatchlistRepository:
        return self.uow.watchlist

    @property
    def memories(self) -> MemoryRepository:
        return self.uow.memories

    @property
    def alerts(self):
        return self.uow.alerts

    @property
    def documents(self):
        return self.uow.documents

    @property
    def jobs(self):
        return self.uow.jobs

    @property
    def profiles(self):
        return self.uow.profiles

    @property
    def user(self):
        return self.uow.users

    @property
    def integrations(self):
        return self.uow.integrations

    @property
    def oauth_flows(self):
        return self.uow.oauth_flows


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


async def _get_market_news(ctx: ToolContext, args: dict[str, Any]) -> str:
    if ctx.finnhub is None:
        return json.dumps({"error": "market data is not configured"})
    limit = int(args.get("limit", 8))
    items = await ctx.finnhub.general_news(limit=limit)
    if not items:
        return json.dumps({"error": "no news available right now"})
    return json.dumps(items)


async def _get_company_news(ctx: ToolContext, args: dict[str, Any]) -> str:
    if ctx.finnhub is None:
        return json.dumps({"error": "market data is not configured"})
    symbol = str(args.get("symbol", "")).upper()
    if not symbol:
        return json.dumps({"error": "symbol is required"})
    limit = int(args.get("limit", 8))
    items = await ctx.finnhub.company_news(symbol, limit=limit)
    if not items:
        return json.dumps({"error": f"no recent news for {symbol}", "symbol": symbol})
    return json.dumps({"symbol": symbol, "news": items})


async def _get_company_earnings(ctx: ToolContext, args: dict[str, Any]) -> str:
    if ctx.finnhub is None:
        return json.dumps({"error": "market data is not configured"})
    symbol = str(args.get("symbol", "")).upper()
    if not symbol:
        return json.dumps({"error": "symbol is required"})
    item = await ctx.finnhub.earnings(symbol)
    if not item:
        return json.dumps({"error": f"no earnings data for {symbol}", "symbol": symbol})
    return json.dumps({"symbol": symbol, "earnings": item})


async def _get_market_indices(ctx: ToolContext, args: dict[str, Any]) -> str:
    if ctx.indices is None:
        return json.dumps({"error": "index data is not configured"})
    try:
        indices = await ctx.indices.fetch()
    except Exception as exc:  # noqa: BLE001 - surface provider errors to the model
        return json.dumps({"error": str(exc)})
    if not indices:
        return json.dumps({"error": "no index data available right now"})
    return json.dumps({"indices": indices, "count": len(indices)})


async def _create_price_alert(ctx: ToolContext, args: dict[str, Any]) -> str:
    symbol = str(args.get("symbol", "")).upper()
    if not symbol:
        return json.dumps({"error": "symbol is required"})
    percent = float(args.get("percent", 5))
    condition = {
        "operator": str(args.get("operator", "abs")),
        "percent": percent,
    }
    if args.get("direction"):
        condition["direction"] = str(args.get("direction"))
    await ctx.alerts.create(ctx.user_id, kind=AlertKind.PRICE, symbol=symbol, condition=condition)
    await ctx.uow.commit()
    return json.dumps({"message": f"created a {percent:g}% price alert for {symbol}"})


async def _create_news_alert(ctx: ToolContext, args: dict[str, Any]) -> str:
    symbol = str(args.get("symbol", "")).upper()
    if not symbol:
        return json.dumps({"error": "symbol is required"})
    condition = {}
    if args.get("keyword"):
        condition["keyword"] = str(args.get("keyword"))
    await ctx.alerts.create(ctx.user_id, kind=AlertKind.NEWS, symbol=symbol, condition=condition)
    await ctx.uow.commit()
    return json.dumps({"message": f"created a news alert for {symbol}"})


async def _create_filing_alert(ctx: ToolContext, args: dict[str, Any]) -> str:
    symbol = str(args.get("symbol", "")).upper()
    if not symbol:
        return json.dumps({"error": "symbol is required"})
    await ctx.alerts.create(
        ctx.user_id,
        kind=AlertKind.FILING,
        symbol=symbol,
        condition={"forms": ["8-K", "10-K", "10-Q"]},
    )
    await ctx.uow.commit()
    return json.dumps({"message": f"created an SEC filing alert for {symbol}"})


async def _list_alerts(ctx: ToolContext, args: dict[str, Any]) -> str:
    alerts = await ctx.alerts.list_enabled(ctx.user_id)
    return json.dumps(
        [
            {
                "id": str(a.id),
                "kind": a.kind.value,
                "symbol": a.symbol,
                "condition": a.condition,
            }
            for a in alerts
        ]
    )


async def _delete_alert(ctx: ToolContext, args: dict[str, Any]) -> str:
    alert_id = str(args.get("alert_id", ""))
    alerts = await ctx.alerts.list_enabled(ctx.user_id)
    target = next((a for a in alerts if str(a.id) == alert_id), None)
    if target is None:
        return json.dumps({"error": "alert not found"})
    await ctx.alerts.update(target, enabled=False)
    await ctx.uow.commit()
    return json.dumps({"message": f"alert {alert_id} removed"})


async def _create_daily_briefing(ctx: ToolContext, args: dict[str, Any]) -> str:
    time = str(args.get("time") or "08:00")
    jobs = await ctx.jobs.list_enabled()
    if any(j.user_id == ctx.user_id and j.job_type == "daily_brief" for j in jobs):
        return json.dumps({"message": "a daily briefing is already scheduled"})
    user = await ctx.user.get_by_id(ctx.user_id)
    tz = user.timezone if user is not None else None
    scope = str(args.get("scope") or "watchlist").lower()
    if scope not in {"watchlist", "interests", "both"}:
        scope = "watchlist"
    await ctx.jobs.create(
        job_type="daily_brief",
        cron_expr=cron_from_local_time(time, tz),
        user_id=ctx.user_id,
        params={"scope": scope},
        timezone=(tz or "UTC"),
    )
    await ctx.profiles.upsert(ctx.user_id, briefing_time=extract_clock_time(time))
    await ctx.uow.commit()
    return json.dumps({"message": f"daily briefing scheduled at {time}"})


async def _create_reminder(ctx: ToolContext, args: dict[str, Any]) -> str:
    text = str(args.get("text") or "").strip()
    if not text:
        return json.dumps({"error": "reminder text is required"})
    time = extract_clock_time(str(args.get("time") or args.get("when") or ""))
    time = time or "09:00"
    user = await ctx.user.get_by_id(ctx.user_id)
    tz = user.timezone if user is not None else None
    cron = cron_from_local_time(time, tz)
    first_run = compute_next_run(cron, after=dt.datetime.now(UTC).astimezone(UTC))
    await ctx.jobs.create(
        job_type="reminder",
        cron_expr=cron,
        user_id=ctx.user_id,
        params={"text": text, "once": bool(args.get("once", False))},
        timezone=(tz or "UTC"),
        next_run_at=first_run,
    )
    await ctx.uow.commit()
    return json.dumps({"message": f"reminder set for {time}: {text}"})


async def _get_document_contents(ctx: ToolContext, args: dict[str, Any]) -> str:
    docs = await ctx.documents.list_for_user(ctx.user_id, limit=3)
    if not docs:
        return json.dumps({"error": "no uploaded documents found"})
    index = int(args.get("index", 0))
    doc = docs[index] if 0 <= index < len(docs) else None
    if doc is None:
        return json.dumps({"error": "document index out of range"})
    text = (doc.doc_meta or {}).get("extracted_text") or ""
    return json.dumps(
        {
            "filename": doc.filename,
            "kind": (doc.doc_meta or {}).get("kind"),
            "status": doc.status.value,
            "text": text[:12_000],
        }
    )


async def _link_google_sheet(ctx: ToolContext, args: dict[str, Any]) -> str:
    from app.domain.enums import IntegrationProvider

    url = str(args.get("url") or "").strip()
    if not url:
        return json.dumps({"error": "a Google Sheets URL is required"})
    link = await ctx.integrations.upsert(
        ctx.user_id,
        provider=IntegrationProvider.SHEETS,
        access_token=url,
        scopes=["read"],
    )
    await ctx.uow.commit()
    return json.dumps(
        {"message": "Google Sheet linked; you can query it now", "sheet_id": str(link.id)}
    )


async def _unlink_google_sheet(ctx: ToolContext, args: dict[str, Any]) -> str:
    from app.domain.enums import IntegrationProvider

    link = await ctx.integrations.get_by_provider(ctx.user_id, IntegrationProvider.SHEETS)
    if link is None:
        return json.dumps({"error": "no linked Google Sheet"})
    await ctx.integrations.delete(link)
    await ctx.uow.commit()
    return json.dumps({"message": "Google Sheet unlinked"})


async def _read_google_sheet(ctx: ToolContext, args: dict[str, Any]) -> str:
    from app.domain.enums import IntegrationProvider
    from app.infrastructure.providers.google_sheets import sheet_id_from_url

    url = str(args.get("url") or "").strip()
    if not url:
        link = await ctx.integrations.get_by_provider(ctx.user_id, IntegrationProvider.SHEETS)
        if link is not None:
            url = link.access_token
    if not url:
        return json.dumps(
            {
                "error": (
                    "no Google Sheet URL given and none linked. Send a sheet "
                    "URL or link one with link_google_sheet."
                )
            }
        )
    if ctx.google_sheets is None:
        return json.dumps({"error": "Google Sheets is not configured"})
    sheet_id = sheet_id_from_url(url)
    if sheet_id is None:
        return json.dumps({"error": "could not find a Google Sheet id in that URL"})
    try:
        rows = await ctx.google_sheets.fetch_rows(url, max_rows=200)
    except Exception as exc:  # noqa: BLE001 - surface provider errors to the model
        return json.dumps({"error": str(exc)})
    if not rows:
        return json.dumps({"error": "that sheet returned no rows", "sheet_id": sheet_id})
    return json.dumps({"sheet_id": sheet_id, "row_count": len(rows), "rows": rows[:20]})


# --- Google OAuth helpers (Gmail / Calendar / Drive) -------------------------


class _GoogleNotConnected(RuntimeError):
    def __init__(self, provider: IntegrationProvider) -> None:
        self.provider = provider
        super().__init__(f"{provider.value} is not connected")


async def _linked_token(
    ctx: ToolContext, provider: IntegrationProvider, *, force_refresh: bool = False
) -> str:
    """Returns a valid access token for the provider, refreshing under a
    per-user lock when expired. One in-flight refresh per user/provider."""
    link = await ctx.integrations.get_by_provider(ctx.user_id, provider)
    if link is None:
        raise _GoogleNotConnected(provider)
    if ctx.google_oauth is None or not ctx.google_oauth.configured:
        raise GoogleOAuthError(
            "Google sign-in is not configured on this server", kind="not_configured"
        )
    async with ctx.google_oauth.lock_for(ctx.user_id):
        fresh = await ctx.integrations.get_by_provider(ctx.user_id, provider)
        if fresh is None:
            raise _GoogleNotConnected(provider)
        expires_in = fresh.expires_at is None or _as_utc(fresh.expires_at) <= dt.datetime.now(
            UTC
        ) + dt.timedelta(seconds=60)
        if expires_in or force_refresh:
            if not fresh.refresh_token:
                raise GoogleOAuthError(
                    "Google needs reconnecting: the stored token can no longer be refreshed",
                    kind="invalid_grant",
                )
            bundle = await ctx.google_oauth.refresh_access_token(fresh.refresh_token)
            await ctx.integrations.upsert(
                ctx.user_id,
                provider=provider,
                access_token=bundle.access_token,
                refresh_token=fresh.refresh_token,
                scopes=fresh.scopes,
                expires_at=dt.datetime.now(UTC) + dt.timedelta(seconds=bundle.expires_in),
            )
            await ctx.uow.commit()
            return bundle.access_token
        return fresh.access_token


async def _google_call(ctx: ToolContext, provider: IntegrationProvider, fn: Any) -> Any:
    """Runs `fn(token)`, refreshing once if the API rejects the token (401)."""
    try:
        token = await _linked_token(ctx, provider)
        return await fn(token)
    except GoogleTokenExpiredError:
        token = await _linked_token(ctx, provider, force_refresh=True)
        return await fn(token)


def _google_failure(exc: Exception) -> str:
    """Maps connector errors to friendly JSON without leaking token data."""
    if isinstance(exc, _GoogleNotConnected):
        return json.dumps(
            {
                "error": (
                    f"Your {exc.provider.value} is not connected. Ask the user "
                    "to connect Google first, or call connect_google."
                )
            }
        )
    if isinstance(exc, GoogleOAuthError):
        if exc.kind == "invalid_grant":
            return json.dumps(
                {
                    "error": (
                        "Google access was revoked or expired. Ask the user to "
                        "reconnect with connect_google."
                    )
                }
            )
        return json.dumps({"error": "Google sign-in is not available right now"})
    if isinstance(exc, GoogleApiError):
        return json.dumps({"error": f"Google API error: {exc}"})
    return json.dumps({"error": f"A Google connector error occurred ({type(exc).__name__}: {exc})"})


def _as_utc(value: dt.datetime) -> dt.datetime:
    """SQLite returns naive datetimes from DateTime(timezone=True); treat them as UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _parse_when(when: str | None, duration_min: int) -> tuple[dt.datetime, dt.datetime] | None:
    """Parses natural meeting times ('tomorrow 10:30', '14:00', ISO) to UTC."""
    text = (when or "").strip().lower()
    if not text:
        return None
    try:
        start = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        return start, start + dt.timedelta(minutes=max(duration_min, 5))
    except ValueError:
        pass
    match = re.search(r"(\d{1,2})[:.](\d{2})", text)
    hour, minute = (int(match.group(1)), int(match.group(2))) if match else (9, 0)
    now = dt.datetime.now(UTC)
    day = now.date()
    if "tomorrow" in text:
        day = now.date() + dt.timedelta(days=1)
    start = dt.datetime.combine(day, dt.time(hour, minute), tzinfo=UTC)
    return start, start + dt.timedelta(minutes=max(duration_min, 5))


# --- Google OAuth handlers ----------------------------------------------------


async def _connect_google(ctx: ToolContext, args: dict[str, Any]) -> str:
    if ctx.google_oauth is None or not ctx.google_oauth.configured:
        return json.dumps({"error": "Google sign-in is not configured on this server"})
    if not ctx.public_base_url:
        return json.dumps({"error": "Google sign-in is not configured on this server"})
    existing = await ctx.integrations.get_by_provider(ctx.user_id, IntegrationProvider.GMAIL)
    if existing is not None:
        return json.dumps({"message": "Google is already connected (Gmail, Calendar, and Drive)."})
    if ctx.chat_id is None:
        return json.dumps({"error": "cannot start Google sign-in without a chat id"})
    state = generate_state()
    verifier = generate_verifier()
    await ctx.oauth_flows.create(
        state=state,
        user_id=ctx.user_id,
        chat_id=ctx.chat_id,
        code_verifier=verifier,
        expires_at=dt.datetime.now(UTC) + dt.timedelta(minutes=10),
    )
    ctx.oauth_connect_url = f"{ctx.public_base_url.rstrip('/')}/oauth/google/start?state={state}"
    await ctx.uow.commit()
    return json.dumps(
        {
            "message": (
                "Tap the button below to connect your Google account — this "
                "grants access to Gmail, Calendar, and Drive (read-only, "
                "except calendar events)."
            )
        }
    )


async def _disconnect_google(ctx: ToolContext, args: dict[str, Any]) -> str:
    removed: list[str] = []
    for provider in (
        IntegrationProvider.GMAIL,
        IntegrationProvider.CALENDAR,
        IntegrationProvider.DRIVE,
    ):
        link = await ctx.integrations.get_by_provider(ctx.user_id, provider)
        if link is None:
            continue
        if ctx.google_oauth is not None and link.refresh_token:
            with contextlib.suppress(GoogleOAuthError):
                # still unlink locally even if remote revoke failed
                await ctx.google_oauth.revoke(link.refresh_token)
        await ctx.integrations.delete(link)
        removed.append(provider.value)
    await ctx.uow.commit()
    if not removed:
        return json.dumps({"message": "There is no Google connection to disconnect."})
    return json.dumps({"message": f"Google disconnected ({', '.join(removed)})"})


async def _search_emails(ctx: ToolContext, args: dict[str, Any]) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "a search query is required"})
    max_results = min(int(args.get("max_results", 15)), 50)

    async def run(token: str) -> str:
        gmail = GmailClient(token, http=ctx.google_http)
        results = await gmail.search(query, max_results=max_results)
        if not results:
            return json.dumps({"query": query, "emails": []})
        messages = []
        for item in results[:max_results]:
            try:
                messages.append(await gmail.get_message(item["id"]))
            except GoogleApiError:
                continue
        return json.dumps({"query": query, "emails": messages})

    try:
        return await _google_call(ctx, IntegrationProvider.GMAIL, run)
    except Exception as exc:  # noqa: BLE001 - map connector failures for the model
        return _google_failure(exc)


async def _find_calendar_events(ctx: ToolContext, args: dict[str, Any]) -> str:
    days = max(1, min(int(args.get("days", 7)), 30))

    async def run(token: str) -> str:
        calendar = CalendarClient(token, http=ctx.google_http)
        events = await calendar.list_events(days=days, max_results=30)
        return json.dumps({"events": events, "count": len(events)})

    try:
        return await _google_call(ctx, IntegrationProvider.CALENDAR, run)
    except Exception as exc:  # noqa: BLE001
        return _google_failure(exc)


async def _schedule_meeting(ctx: ToolContext, args: dict[str, Any]) -> str:
    summary = str(args.get("summary") or "").strip()
    if not summary:
        return json.dumps({"error": "a meeting title is required"})
    duration_min = max(5, int(args.get("duration_min", 60)))
    parsed = _parse_when(args.get("when"), duration_min)
    if parsed is None:
        return json.dumps(
            {"error": "I need a meeting time like 'tomorrow 10:30' or an ISO timestamp"}
        )
    start, end = parsed
    attendees = [str(a).strip() for a in (args.get("attendees") or []) if str(a).strip()]

    async def run(token: str) -> str:
        calendar = CalendarClient(token, http=ctx.google_http)
        event = await calendar.create_event(
            summary=summary,
            start=start,
            end=end,
            description=args.get("description"),
            attendees=attendees or None,
        )
        return json.dumps(
            {
                "event_id": event.get("id"),
                "summary": event.get("summary"),
                "start": (event.get("start") or {}).get("dateTime"),
                "end": (event.get("end") or {}).get("dateTime"),
                "link": event.get("htmlLink"),
            }
        )

    try:
        return await _google_call(ctx, IntegrationProvider.CALENDAR, run)
    except Exception as exc:  # noqa: BLE001
        return _google_failure(exc)


async def _read_drive_doc(ctx: ToolContext, args: dict[str, Any]) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "a search query is required"})

    async def run(token: str) -> str:
        drive = DriveClient(token, http=ctx.google_http)
        files = await drive.search(query, max_results=5)
        if not files:
            return json.dumps({"query": query, "matches": [], "error": "no files found in Drive"})
        chosen = files[0]
        raw = await drive.download(
            chosen["id"],
            mime_type=chosen.get("mime_type"),
            filename=chosen.get("name") or "drive_file",
        )
        if ctx.media_pipeline is None:
            return json.dumps(
                {
                    "matches": files,
                    "error": "document analysis is not available right now",
                }
            )
        result = await ctx.media_pipeline.process(
            file_id="",
            mime_type=chosen.get("mime_type"),
            filename=chosen.get("name"),
            data=FileData(
                raw=raw,
                mime_type=chosen.get("mime_type"),
                filename=chosen.get("name") or "drive_file",
            ),
        )
        if result.error is not None or result.document is None:
            return json.dumps(
                {
                    "matches": files,
                    "error": result.error or "could not read that file",
                    "error_code": result.error_code,
                }
            )
        return json.dumps(
            {
                "query": query,
                "matches": files,
                "chosen": chosen,
                "kind": result.document.kind.value,
                "content": (result.content or result.document.text)[:8_000],
            }
        )

    try:
        return await _google_call(ctx, IntegrationProvider.DRIVE, run)
    except Exception as exc:  # noqa: BLE001
        return _google_failure(exc)


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
    Tool(
        name="get_market_news",
        description="Get the latest general market news headlines with sources.",
        parameters={"limit": {"type": "integer", "description": "Max headlines (default 8)"}},
        handler=_get_market_news,
    ),
    Tool(
        name="get_market_indices",
        description=(
            "Get current levels of the major US indices (S&P 500, Dow Jones, "
            "Nasdaq) with change and change percent."
        ),
        parameters={},
        handler=_get_market_indices,
    ),
    Tool(
        name="get_company_news",
        description="Get recent news headlines for a company symbol (e.g. AAPL).",
        parameters={
            "symbol": {"type": "string", "description": "US stock ticker symbol"},
            "limit": {"type": "integer", "description": "Max headlines (default 8)"},
        },
        required=["symbol"],
        handler=_get_company_news,
    ),
    Tool(
        name="get_company_earnings",
        description=(
            "Get the latest earnings event (date, estimates, actuals) for a company symbol."
        ),
        parameters={"symbol": {"type": "string", "description": "US stock ticker symbol"}},
        required=["symbol"],
        handler=_get_company_earnings,
    ),
    Tool(
        name="create_price_alert",
        description=(
            "Create an alert that notifies the user when a stock moves more than a "
            "percent in a day. operator: abs (any direction), gte, lte."
        ),
        parameters={
            "symbol": {"type": "string", "description": "US stock ticker symbol"},
            "percent": {"type": "number", "description": "Percent threshold, default 5"},
            "operator": {
                "type": "string",
                "description": "abs|gte|lte, default abs",
            },
            "direction": {
                "type": "string",
                "description": "Optional: up|down for directional triggers",
            },
        },
        required=["symbol"],
        handler=_create_price_alert,
    ),
    Tool(
        name="create_news_alert",
        description="Create an alert that notifies on Reuters-hit news for a company symbol.",
        parameters={
            "symbol": {"type": "string", "description": "US stock ticker symbol"},
            "keyword": {"type": "string", "description": "Optional keyword to match"},
        },
        required=["symbol"],
        handler=_create_news_alert,
    ),
    Tool(
        name="create_filing_alert",
        description=(
            "Create an alert that notifies when a company files an SEC form (8-K, 10-K, 10-Q)."
        ),
        parameters={"symbol": {"type": "string", "description": "US stock ticker symbol"}},
        required=["symbol"],
        handler=_create_filing_alert,
    ),
    Tool(
        name="list_alerts",
        description="List the user's active alerts.",
        parameters={},
        handler=_list_alerts,
    ),
    Tool(
        name="delete_alert",
        description="Delete/disable an alert by its alert_id (see list_alerts).",
        parameters={"alert_id": {"type": "string", "description": "Alert id to remove"}},
        required=["alert_id"],
        handler=_delete_alert,
    ),
    Tool(
        name="create_daily_briefing",
        description=(
            "Schedule (or reschedule) the user's daily morning briefing. "
            "Pass a 24h local time like '08:00'. scope: watchlist (default), "
            "interests (topics like AI/semiconductors/tech), or both."
        ),
        parameters={
            "time": {"type": "string", "description": "HH:MM local time, default 08:00"},
            "scope": {
                "type": "string",
                "description": "watchlist|interests|both, default watchlist",
            },
        },
        handler=_create_daily_briefing,
    ),
    Tool(
        name="create_reminder",
        description=(
            "Schedule a reminder. Use when the user says 'remind me'. "
            "Pass 'time' like '09:00' or 'when', the text, and once=true for a single reminder."
        ),
        parameters={
            "text": {"type": "string", "description": "Reminder text"},
            "time": {"type": "string", "description": "HH:MM local time to remind"},
            "once": {"type": "boolean", "description": "True if this should fire only once"},
        },
        required=["text"],
        handler=_create_reminder,
    ),
    Tool(
        name="get_document_contents",
        description=(
            "Re-read the text of a previously uploaded document for follow-up questions "
            "(index 0 = most recent)."
        ),
        parameters={"index": {"type": "integer", "description": "0 = most recent document"}},
        handler=_get_document_contents,
    ),
    Tool(
        name="link_google_sheet",
        description=(
            "Remember a Google Sheets URL for the user so they can be queried later "
            "without resending the link."
        ),
        parameters={"url": {"type": "string", "description": "Google Sheets share URL"}},
        required=["url"],
        handler=_link_google_sheet,
    ),
    Tool(
        name="unlink_google_sheet",
        description="Forget the user's linked Google Sheet.",
        parameters={},
        handler=_unlink_google_sheet,
    ),
    Tool(
        name="read_google_sheet",
        description=(
            "Read rows from a Google Sheet. Uses the linked sheet if no URL is given. "
            "Useful to pull a model portfolio, pipeline resigning data, or spreadsheets "
            "the user keeps in Drive."
        ),
        parameters={"url": {"type": "string", "description": "Optional Google Sheets URL"}},
        handler=_read_google_sheet,
    ),
    Tool(
        name="connect_google",
        description=(
            "Start connecting the user's Google account (Gmail, Calendar, Drive). "
            "Use when the user wants to search emails, manage meetings, or read "
            "Drive files, and nothing is connected yet. A button appears for sign-in."
        ),
        parameters={},
        handler=_connect_google,
    ),
    Tool(
        name="disconnect_google",
        description=(
            "Disconnect the user's Google account: revokes access and removes "
            "stored credentials for Gmail, Calendar, and Drive."
        ),
        parameters={},
        handler=_disconnect_google,
    ),
    Tool(
        name="search_emails",
        description=(
            "Search the user's Gmail and return matching messages (from, subject, "
            "date, body excerpt). Use a Gmail-style query like 'subject:tesla' or "
            "plain terms like 'tesla earnings'."
        ),
        parameters={
            "query": {"type": "string", "description": "Gmail search query"},
            "max_results": {"type": "integer", "description": "Max messages (default 15)"},
        },
        required=["query"],
        handler=_search_emails,
    ),
    Tool(
        name="find_calendar_events",
        description=(
            "List the user's upcoming Google Calendar events (default: next 7 days) "
            "for meeting preparation."
        ),
        parameters={"days": {"type": "integer", "description": "Lookahead days (default 7)"}},
        handler=_find_calendar_events,
    ),
    Tool(
        name="schedule_meeting",
        description=(
            "Create a Google Calendar event for the user. Pass a title, a natural "
            "time like 'tomorrow 10:30' or an ISO timestamp, and optional attendees."
        ),
        parameters={
            "summary": {"type": "string", "description": "Meeting title"},
            "when": {"type": "string", "description": "e.g. 'tomorrow 10:30' or ISO"},
            "duration_min": {"type": "integer", "description": "Duration in minutes (default 60)"},
            "description": {"type": "string", "description": "Optional agenda/notes"},
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional emails",
            },
        },
        required=["summary", "when"],
        handler=_schedule_meeting,
    ),
    Tool(
        name="read_drive_doc",
        description=(
            "Search the user's Google Drive for a file (PDF, spreadsheet, text, "
            "Google Doc/Sheet) and read/summarize its contents."
        ),
        parameters={"query": {"type": "string", "description": "Drive search terms or filename"}},
        required=["query"],
        handler=_read_drive_doc,
    ),
]


def default_registry() -> ToolRegistry:
    return ToolRegistry(DEFAULT_TOOLS)


__all__ = ["Tool", "ToolContext", "ToolRegistry", "DEFAULT_TOOLS", "default_registry"]
