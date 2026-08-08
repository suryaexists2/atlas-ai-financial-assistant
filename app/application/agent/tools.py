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
        detail = f" ({exc.status})" if exc.status else ""
        return json.dumps({"error": f"Google API error{detail}: {exc}"})
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
    raw_attendees = args.get("attendees") or []
    if isinstance(raw_attendees, str):
        if raw_attendees.strip():
            try:
                raw_attendees = json.loads(raw_attendees)
            except (json.JSONDecodeError, TypeError):
                raw_attendees = [raw_attendees]
        else:
            raw_attendees = []
    attendees = []
    for a in raw_attendees:
        if isinstance(a, dict):
            a = a.get("email") or a.get("value") or ""
        a = str(a).strip()
        if a:
            attendees.append(a)

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


def _drive_search_query(text: str) -> str:
    """Converts free-text terms into a valid Drive `q` parameter; passes
    through queries that already contain Drive operators."""
    lowered = text.lower()
    if any(op in lowered for op in (" contains ", " in parents", " and ", " or ", "mimetype=")):
        return text
    return f"name contains '{text.replace(chr(39), chr(92) + chr(39))}'"


async def _read_drive_doc(ctx: ToolContext, args: dict[str, Any]) -> str:
    raw_query = str(args.get("query") or "").strip()
    if not raw_query:
        return json.dumps({"error": "a search query is required"})
    query = _drive_search_query(raw_query)

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


_TOOL_PARAM_TYPES = {
    "string": {"type": "string"},
    "integer": {"type": "integer"},
    "number": {"type": "number"},
    "boolean": {"type": "boolean"},
    "object": {"type": "object"},
    "strlist": {"type": "array", "items": {"type": "string"}},
}

# (name, description, {param: type}, required). Kept data-driven so the schemas
# stay small for the free-tier Groq TPM window while every tool stays available.
_TOOL_SPECS: list[tuple[str, str, dict[str, str], tuple[str, ...]]] = [
    (
        'get_market_quote',
        'Quote (price, change, high/low) for a US ticker.',
        {'symbol': 'string'},
        ('symbol',),
    ),
    (
        'get_company_profile',
        'Company profile (name, exchange, industry, market cap).',
        {'symbol': 'string'},
        ('symbol',),
    ),
    (
        'get_company_filings',
        'Recent SEC filings (10-K, 10-Q, 8-K default).',
        {'symbol': 'string', 'form_types': 'strlist', 'limit': 'integer'},
        ('symbol',),
    ),
    (
        'list_watchlist',
        "List the user's watchlist.",
        {},
        (),
    ),
    (
        'add_to_watchlist',
        'Add a symbol to the watchlist.',
        {'symbol': 'string', 'name': 'string', 'sector': 'string'},
        ('symbol',),
    ),
    (
        'remove_from_watchlist',
        'Remove a symbol from the watchlist.',
        {'symbol': 'string'},
        ('symbol',),
    ),
    (
        'save_memory',
        "Remember a fact. key 'user_profile' for traits; 'interest:<topic>' for interests.",
        {'memory_key': 'string', 'summary': 'string', 'value': 'object', 'confidence': 'number'},
        ('memory_key', 'summary'),
    ),
    (
        'list_memories',
        'List stored memories.',
        {'limit': 'integer'},
        (),
    ),
    (
        'get_market_news',
        'Latest general market news headlines.',
        {'limit': 'integer'},
        (),
    ),
    (
        'get_market_indices',
        'Major US indices (S&P 500, Dow, Nasdaq) levels and change.',
        {},
        (),
    ),
    (
        'get_company_news',
        'Recent news for a ticker.',
        {'symbol': 'string', 'limit': 'integer'},
        ('symbol',),
    ),
    (
        'get_company_earnings',
        'Latest earnings event for a ticker.',
        {'symbol': 'string'},
        ('symbol',),
    ),
    (
        'create_price_alert',
        'Alert when a stock moves over X percent in a day (operator abs|gte|lte).',
        {'symbol': 'string', 'percent': 'number', 'operator': 'string', 'direction': 'string'},
        ('symbol',),
    ),
    (
        'create_news_alert',
        'Alert on news for a ticker.',
        {'symbol': 'string', 'keyword': 'string'},
        ('symbol',),
    ),
    (
        'create_filing_alert',
        'Alert when a ticker files 8-K/10-K/10-Q.',
        {'symbol': 'string'},
        ('symbol',),
    ),
    (
        'list_alerts',
        'List active alerts.',
        {},
        (),
    ),
    (
        'delete_alert',
        'Remove an alert by alert_id (from list_alerts).',
        {'alert_id': 'string'},
        ('alert_id',),
    ),
    (
        'create_daily_briefing',
        "Schedule the morning briefing. time like '08:00'; scope watchlist|interests|both.",
        {'time': 'string', 'scope': 'string'},
        (),
    ),
    (
        'create_reminder',
        "Schedule a reminder; time like '09:00', text, once=true.",
        {'text': 'string', 'time': 'string', 'once': 'boolean'},
        ('text',),
    ),
    (
        'get_document_contents',
        'Re-read an uploaded document (index 0 = most recent).',
        {'index': 'integer'},
        (),
    ),
    (
        'link_google_sheet',
        'Remember a Sheets URL to query later.',
        {'url': 'string'},
        ('url',),
    ),
    (
        'unlink_google_sheet',
        'Forget the linked sheet.',
        {},
        (),
    ),
    (
        'read_google_sheet',
        'Read rows from a Google Sheet (linked one if no URL).',
        {'url': 'string'},
        (),
    ),
    (
        'connect_google',
        'Start Google sign-in (Gmail, Calendar, Drive); a button appears.',
        {},
        (),
    ),
    (
        'disconnect_google',
        'Disconnect Google (Gmail, Calendar, Drive).',
        {},
        (),
    ),
    (
        'search_emails',
        "Search Gmail (e.g. 'subject:tesla' or 'tesla earnings'); returns matching messages.",
        {'query': 'string', 'max_results': 'integer'},
        ('query',),
    ),
    (
        'find_calendar_events',
        'Upcoming Google Calendar events (default 7 days).',
        {'days': 'integer'},
        (),
    ),
    (
        'schedule_meeting',
        "Create a Calendar event: title, natural time ('tomorrow 10:30' or ISO), attendees.",
        {'summary': 'string', 'when': 'string', 'duration_min': 'integer', 'description': 'string', 'attendees': 'strlist'},
        ('summary', 'when'),
    ),
    (
        'read_drive_doc',
        'Search Drive for a file and read it.',
        {'query': 'string'},
        ('query',),
    ),
]

_UNUSUAL_HANDLERS = {"get_market_quote": _get_quote, "get_company_filings": _get_filings}


def _handler_for(name: str) -> ToolHandler:
    handler = _UNUSUAL_HANDLERS.get(name) or globals().get(f"_{name}")
    assert handler is not None, f"missing handler for {name}"
    return handler


DEFAULT_TOOLS: list[Tool] = [
    Tool(
        name=name,
        description=description,
        parameters={param: dict(_TOOL_PARAM_TYPES[ptype]) for param, ptype in params.items()},
        required=list(required),
        handler=_handler_for(name),
    )
    for name, description, params, required in _TOOL_SPECS
]


def default_registry() -> ToolRegistry:
    return ToolRegistry(DEFAULT_TOOLS)


__all__ = ["Tool", "ToolContext", "ToolRegistry", "DEFAULT_TOOLS", "default_registry"]
