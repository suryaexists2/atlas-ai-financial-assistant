"""Morning briefing handler — the assistant's main proactive touchpoint.

Composes a short brief from the user's watchlist quotes plus current market
news, explains why it matters, and delivers via the outbox. When no LLM is
configured or the LLM is unavailable, it falls back to a clean deterministic
summary. Stays silent when there is genuinely nothing to report.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from app.application.intelligence import IntelligenceContext, enqueue_for_user
from app.core.logging import get_logger
from app.infrastructure.db.uow import UnitOfWork

logger = get_logger(__name__)

_BRIEF_SYSTEM = (
    "You are Atlas, writing a personalized morning market briefing for a "
    "finance professional inside Telegram.\n"
    "Rules:\n"
    "- Use ONLY the facts in the DATA block below. Never invent prices, dates, or headlines.\n"
    "- Be concise, 120-180 words, 2-3 short paragraphs. No markdown headers.\n"
    "- Explain why the moves matter for the user's watchlist and interests.\n"
    "- If a fact is missing, say nothing about it.\n"
)

_MAX_WATCHLIST_QUOTES = 12
_MAX_NEWS_LINES = 5
_MAX_INTEREST_NEWS = 5


async def daily_brief(uow: UnitOfWork, job, ctx: IntelligenceContext) -> None:
    if job.user_id is None:
        return
    user = await uow.users.get_by_id(job.user_id)
    if user is None:
        return

    scope = (job.params or {}).get("scope", "watchlist")
    interests = await _gather_interests(uow, job.user_id)

    watchlist = await uow.watchlist.list_active(job.user_id)
    quotes: list[dict[str, Any]] = []
    if ctx.finnhub is not None:
        for item in watchlist[:_MAX_WATCHLIST_QUOTES]:
            try:
                quote = await ctx.finnhub.quote(item.symbol)
            except Exception:  # noqa: BLE001 - a single missing quote is not fatal
                continue
            if quote and quote.get("c") is not None:
                quotes.append(
                    {
                        "symbol": item.symbol,
                        "name": item.name or item.symbol,
                        "c": quote.get("c"),
                        "d": quote.get("d"),
                        "dp": quote.get("dp"),
                        "pc": quote.get("pc"),
                    }
                )

    news: list[dict[str, Any]] = []
    if ctx.finnhub is not None:
        try:
            news = await ctx.finnhub.general_news(limit=16)
        except Exception:  # noqa: BLE001
            news = []
    news = news[:8]

    # Interest-relevant headlines (e.g. "AI, semiconductors, technology") when the
    # user asked for interest coverage — spec: "Create a daily morning briefing
    # covering AI, semiconductor, and technology stocks."
    interest_news: list[dict[str, Any]] = []
    if "interests" in scope and interests:
        interest_news = _match_interest_news(news, interests)[:_MAX_INTEREST_NEWS]

    # Nothing worth reporting? Stay silent rather than sending noise.
    if not quotes and not news and not interest_news:
        logger.info("briefing_silent_nothing_to_report", user_id=str(job.user_id))
        return

    content = await _compose(watchlist, quotes, news, ctx, interests, interest_news)
    await enqueue_for_user(uow, job, content)


async def _gather_interests(uow: UnitOfWork, user_id: Any) -> list[str]:
    """Collects the user's topics of interest (AI, semiconductor, macro, ...)
    from the profile and the durable memory written by onboarding."""
    topics: list[str] = []
    profile = await uow.profiles.get_by_user_id(user_id)
    if profile is not None and profile.interests:
        for item in profile.interests:
            if isinstance(item, str) and item.strip():
                topics.append(item.strip())
    memories = await uow.memories.list_active(user_id, limit=50)
    for memory in memories:
        if memory.memory_key == "user_interests":
            value = memory.value or {}
            for item in value.get("interests") or []:
                if isinstance(item, str) and item.strip():
                    topics.append(item.strip())
    seen: set[str] = set()
    unique = []
    for topic in topics:
        key = topic.lower()
        if key not in seen:
            seen.add(key)
            unique.append(topic)
    return unique[:8]


def _match_interest_news(news: list[dict[str, Any]], interests: list[str]) -> list[dict[str, Any]]:
    lowered = [i.lower() for i in interests]
    matched: list[dict[str, Any]] = []
    for item in news:
        text = f"{item.get('headline') or ''} {item.get('summary') or ''}".lower()
        if any(token in text for token in lowered):
            matched.append(item)
    return matched


async def _compose(
    watchlist: list[Any],
    quotes: list[dict[str, Any]],
    news: list[dict[str, Any]],
    ctx: IntelligenceContext,
    interests: list[str] | None = None,
    interest_news: list[dict[str, Any]] | None = None,
) -> str:
    data = _build_data_block(quotes, news, interest_news or [])
    if ctx.gateway is not None:
        try:
            response = await ctx.gateway.complete(
                [
                    {"role": "system", "content": _BRIEF_SYSTEM},
                    {
                        "role": "user",
                        "content": "DATA:\n" + data,
                    },
                ],
                max_tokens=300,
                temperature=0.4,
            )
            if response.content and response.content.strip():
                return response.content.strip()
        except Exception:  # noqa: BLE001 - fall back to the deterministic summary
            logger.warning("briefing_llm_failed_falling_back")
    return _deterministic_summary(watchlist, quotes, news, interests or [], interest_news or [])


def _build_data_block(
    quotes: list[dict[str, Any]],
    news: list[dict[str, Any]],
    interest_news: list[dict[str, Any]] | None = None,
) -> str:
    lines: list[str] = []
    if quotes:
        lines.append("Watchlist quotes:")
        for q in quotes:
            change = q.get("dp")
            change_str = f"{change:+.2f}%" if isinstance(change, (int, float)) else "n/a"
            prev = q.get("pc")
            prev_str = f"{float(prev):,.2f}" if isinstance(prev, (int, float)) else "n/a"
            lines.append(
                f"- {q['symbol']} ({q['name']}): ${float(q['c']):,.2f} "
                f"({change_str}, prev close ${prev_str})"
            )
    interest_news = interest_news or []
    if interest_news:
        lines.append("Interest-relevant news:")
        for item in interest_news:
            headline = (item.get("headline") or "").strip()
            source = (item.get("source") or "").strip()
            if headline:
                lines.append(f"- News | {source}: {headline}")
    if news:
        lines.append("Today's market headlines:")
        for item in news[:_MAX_NEWS_LINES]:
            headline = (item.get("headline") or "").strip()
            source = (item.get("source") or "").strip()
            if headline:
                lines.append(f"- News | {source}: {headline}")
    return "\n".join(lines) if lines else "- No fresh data."


def _deterministic_summary(
    watchlist: list[Any],
    quotes: list[dict[str, Any]],
    news: list[dict[str, Any]],
    interests: list[str] | None = None,
    interest_news: list[dict[str, Any]] | None = None,
) -> str:
    date_str = dt.datetime.now(dt.UTC).strftime("%a, %b %d")
    lines = [f"🔔 Morning brief · {date_str}", ""]
    interest_news = interest_news or []
    if interest_news:
        topics = ", ".join(interests or []) or "your interests"
        lines.append(f"Your topics ({topics}):")
        for item in interest_news:
            lines.append(f"• {item.get('headline') or ''}")
        lines.append("")
    if watchlist:
        lines.append("Your watchlist:")
        for q in quotes[:_MAX_WATCHLIST_QUOTES]:
            change = q.get("dp")
            change_str = f"{change:+.2f}%" if isinstance(change, (int, float)) else "n/a"
            lines.append(f"• {q['symbol']} — ${float(q['c']):,.2f} ({change_str})")
        if not quotes:
            lines.append("• quotes unavailable right now")
    lines.append("Prices are spot and may be delayed.")
    return "\n".join(lines)


__all__ = ["_compose", "daily_brief", "_match_interest_news", "_gather_interests"]
