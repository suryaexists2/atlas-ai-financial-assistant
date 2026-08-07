"""Alert evaluators: price, news, and SEC-filing triggers.

Each alert lives in the `alerts` table and is evaluated on a schedule by the
scheduler worker (global cycle jobs). A trigger fires exactly one Telegram
message per user and records `last_fired_at` so a sustained move or repeat
headline cannot spam.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from app.application.intelligence import IntelligenceContext
from app.core.logging import get_logger
from app.domain.enums import AlertKind
from app.infrastructure.db.uow import UnitOfWork

logger = get_logger(__name__)

_PRICE_COOLDOWN = dt.timedelta(hours=4)
_NEWS_COOLDOWN = dt.timedelta(hours=1)
_FILING_COOLDOWN = dt.timedelta(hours=4)
_MAX_ALERTS_PER_CYCLE = 50


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


async def run_price_alerts(uow: UnitOfWork, job, ctx: IntelligenceContext) -> None:
    if ctx.finnhub is None:
        return
    alerts = await uow.alerts.list_enabled()
    fired = 0
    for alert in alerts[:_MAX_ALERTS_PER_CYCLE]:
        if alert.kind is not AlertKind.PRICE or not alert.symbol:
            continue
        if await _within_cooldown(uow, alert, _PRICE_COOLDOWN):
            continue
        try:
            quote = await ctx.finnhub.quote(alert.symbol)
        except Exception:  # noqa: BLE001 - provider hiccup: skip this symbol
            continue
        change = quote.get("dp")
        if not isinstance(change, (int, float)):
            continue
        if _price_triggered(alert.condition, change):
            await _fire(uow, alert, f"🔔 {alert.symbol} moved {change:+.2f}%")
            fired += 1
    if fired:
        logger.info("price_alerts_fired", count=fired)


async def run_news_alerts(uow: UnitOfWork, job, ctx: IntelligenceContext) -> None:
    if ctx.finnhub is None:
        return
    alerts = await uow.alerts.list_enabled()
    fired = 0
    for alert in alerts[:_MAX_ALERTS_PER_CYCLE]:
        if alert.kind is not AlertKind.NEWS or not alert.symbol:
            continue
        if await _within_cooldown(uow, alert, _NEWS_COOLDOWN):
            continue
        try:
            items = await ctx.finnhub.company_news(alert.symbol, limit=5)
        except Exception:  # noqa: BLE001
            continue
        condition = alert.condition or {}
        keyword = condition.get("keyword") or condition.get("text")
        headline = _matching_headline(items, keyword, since=alert.last_fired_at)
        if headline is None:
            continue
        await _fire(uow, alert, f"📰 {alert.symbol}: {headline[:160]}")
        fired += 1
    if fired:
        logger.info("news_alerts_fired", count=fired)


async def run_filing_alerts(uow: UnitOfWork, job, ctx: IntelligenceContext) -> None:
    if ctx.sec is None:
        return
    alerts = await uow.alerts.list_enabled()
    fired = 0
    for alert in alerts[:_MAX_ALERTS_PER_CYCLE]:
        if alert.kind is not AlertKind.FILING or not alert.symbol:
            continue
        if await _within_cooldown(uow, alert, _FILING_COOLDOWN):
            continue
        try:
            filings = await ctx.sec.recent_filings(
                alert.symbol, form_types=["8-K", "10-K", "10-Q"], limit=3
            )
        except Exception:  # noqa: BLE001
            continue
        new_filing = _new_filing(filings, since=alert.last_fired_at)
        if new_filing is None:
            continue
        await _fire(
            uow,
            alert,
            f"📄 {alert.symbol} filed {new_filing['form']} on {new_filing.get('filed_on') or '–'}",
        )
        fired += 1
    if fired:
        logger.info("filing_alerts_fired", count=fired)


def _price_triggered(condition: dict[str, Any] | None, change: float) -> bool:
    condition = condition or {}
    threshold = float(condition.get("percent") or condition.get("threshold") or 5)
    operator = str(condition.get("operator", "abs")).lower()
    if operator in ("gt", ">"):
        return change > threshold
    if operator in ("gte", ">="):
        return change >= threshold
    if operator in ("lt", "<"):
        return change < -threshold
    if operator in ("lte", "<="):
        return change <= -threshold
    return abs(change) >= threshold


def _matching_headline(
    items: list[dict[str, Any]], keyword: str | None, since: dt.datetime | None
) -> str | None:
    for item in items:
        headline = (item.get("headline") or "").strip()
        if not headline:
            continue
        ts = item.get("datetime")
        if since is not None and since.tzinfo is None:
            since = since.replace(tzinfo=dt.UTC)
        if since is not None and isinstance(ts, (int, float)):
            published = dt.datetime.fromtimestamp(ts, dt.UTC)
            if published < since:
                continue
        if keyword:
            haystack = f"{headline} {item.get('summary') or ''}".lower()
            if keyword.lower() not in haystack:
                continue
        return headline
    return None


def _new_filing(filings: list[dict[str, Any]], since: dt.datetime | None) -> dict[str, Any] | None:
    for filing in filings:
        filed_on = filing.get("filed_on")
        if since is not None and since.tzinfo is None:
            since = since.replace(tzinfo=dt.UTC)
        try:
            filed_dt = (
                dt.datetime.strptime(filed_on, "%Y-%m-%d").replace(tzinfo=dt.UTC)
                if filed_on
                else None
            )
        except (TypeError, ValueError):
            filed_dt = None
        if since is not None and filed_dt is not None and filed_dt < since:
            continue
        return filing
    return None


async def _within_cooldown(uow: UnitOfWork, alert, cooldown: dt.timedelta) -> bool:
    if alert.last_fired_at is None:
        return False
    last_fired = alert.last_fired_at
    if last_fired.tzinfo is None:
        last_fired = last_fired.replace(tzinfo=dt.UTC)
    return _now() - last_fired < cooldown


async def _fire(uow: UnitOfWork, alert, text: str) -> None:
    # job.user_id isn't used for alerts; resolve the user directly.
    chat_id = None
    user = await uow.users.get_by_id(alert.user_id)
    if user is not None:
        chat_id = user.telegram_id
    if chat_id is None:
        return
    await uow.outbox.enqueue(
        chat_id=chat_id,
        payload={"type": "text", "text": text, "correlation_id": f"alert:{alert.id}"},
        priority=8,
    )
    await uow.alerts.update(alert, last_fired_at=_now())


__all__ = [
    "run_price_alerts",
    "run_news_alerts",
    "run_filing_alerts",
    "_price_triggered",
    "_matching_headline",
    "_new_filing",
]
