"""Conversational onboarding.

Drives a short, skip-able welcome conversation that learns the user's role,
interests, watchlist and preferred briefing time — one friendly question at a
time, exactly like chatting with an analyst. No forms, no menus.

The engine is deterministic (no LLM): every question is a plain message, and
every reply is parsed with lightweight heuristics. Answers are written into
the user's profile, memory, watchlist and (optionally) the daily briefing job,
so the assistant is personalized from the very first turn.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from app.core.logging import get_logger
from app.domain.enums import OnboardingStatus
from app.infrastructure.db.uow import UnitOfWork

logger = get_logger(__name__)

_SKIP = {"skip", "later", "not now", "nahi", "pass", "none", "no thanks", "nope", "whatever"}
_GREETINGS = {
    "hi", "hello", "hey", "heya", "heyy", "hii", "hlo", "helo", "hellow",
    "good morning", "good afternoon", "good evening", "good night", "morning",
    "evening", "gm", "namaste", "hola", "yo", "sup", "howdy",
    "hi there", "hey there", "hello there", "hi! how are you", "how are you",
}
_ROLES = {
    "investor",
    "analyst",
    "founder",
    "entrepreneur",
    "trader",
    "portfolio manager",
    "fund manager",
    "finance professional",
    "finance",
    "student",
    "advisor",
    "broker",
    "researcher",
    "cfo",
    "consultant",
    "fintech developer",
}
_KNOWN_INTERESTS = {
    "market",
    "markets",
    "stocks",
    "equities",
    "earnings",
    "sec",
    "filings",
    "macro",
    "macroeconomics",
    "economy",
    "crypto",
    "bonds",
    "mutual funds",
    "etf",
    "etfs",
    "options",
    "futures",
    "ipo",
    "funding",
    "startups",
    "venture",
    "ai",
    "artificial intelligence",
    "semiconductors",
    "technology",
    "tech",
    "healthcare",
    "biotech",
    "education",
    "energy",
    "renewables",
    "banking",
    "fintech",
    "real estate",
    "dividends",
    "retirement",
    "index funds",
}
_STOP = {"a", "an", "of", "and", "for", "or", "to", "the", "i", "am", "mostly", "like", "want"}

_SYMBOL_RE = re.compile(r"\b[A-Z]{1,5}(?:[-.][A-Z]{1,5})?\b")
_TIME_RE = re.compile(r"(?i)(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.?m\.?|p\.?m\.?)?")


@dataclass
class OnboardingReply:
    """One turn of the onboarding conversation."""

    text: str | None = None  # message to send the user
    still_onboarding: bool = False  # keep onboarding going next turn
    completed: bool = False  # onboarding finished (or shouldn't run) after this reply
    followed_by_agent: bool = False  # completed and the agent should also answer now


_WELCOME = (
    "Hi! I'm Atlas — your financial assistant.\n"
    "I can pull live quotes, filings and news, read your documents and voice "
    "notes, and keep an eye on the companies you care about.\n\n"
    "A few quick questions make everything personalized — say 'skip' any time "
    "and we'll get straight to work. What best describes you? "
    "(Investor, Analyst, Founder, Student, Finance Professional…)"
)
_ROLE_THEN = "Nice. Which companies, sectors, or markets do you follow? (e.g. Nvidia, AI, energy)"
_INTERESTS_THEN = (
    "Anything you'd like me to monitor? Name tickers, say 'notify me on news' "
    "or alert me on filings, and I'll watch them."
)
_WATCHLIST_THEN = (
    "When would you like your morning briefing? A short summary of your "
    "watchlist and the day's market events — e.g. '8:00' or 'skip'."
)
_BRIEFING_THEN = (
    "Last one: any reminders or alerts you'd like me to keep? For example, "
    "'remind me an hour before Apple's earnings' or 'alert me when AAPL moves "
    "more than 5%'. Or say 'skip'."
)
_CONNECT_THEN = (
    "One optional step: connect your Google account (Gmail, Calendar, Drive)? "
    "Then I can search your emails, schedule meetings, and read Drive files "
    "for you. Say 'skip' — you can connect anytime later by simply asking."
)
_DONE = (
    "You're all set{name}. Just ask — market moves, news, filings, your "
    "documents, or a meeting — and I'll take it from there.\n"
    "I'll send your {brief}."
)


class OnboardingEngine:
    """Stateful, LLM-free onboarding. Call once per incoming message."""

    def __init__(
        self,
        *,
        default_briefing_time: str = "08:00",
        google_connect_available: bool = False,
    ) -> None:
        self._default_briefing_time = default_briefing_time
        self._google_connect_available = google_connect_available

    async def turn(
        self,
        uow: UnitOfWork,
        *,
        user_id: uuid.UUID,
        text: str | None,
        is_media: bool = False,
    ) -> OnboardingReply:
        profile = await uow.profiles.get_by_user_id(user_id)
        if profile is None:
            profile = await uow.profiles.upsert(user_id)
        if profile.onboarding_status == OnboardingStatus.COMPLETED:
            return OnboardingReply(completed=True)

        step = (profile.onboarding_context or {}).get("step", "welcome")

        if is_media:
            await self._complete(uow, user_id)
            return OnboardingReply(
                text=(
                    "Jumping straight in — I'll process files and questions as they land. "
                    "Whenever you're ready, tell me your interests and I'll tailor updates."
                ),
                completed=True,
            )

        if _wants_agent(text):
            await self._complete(uow, user_id)
            return OnboardingReply(completed=True, followed_by_agent=True)

        if step == "welcome":
            if _is_skip(text):
                # Skipping right away: configure the default briefing and hand
                # over to the agent so the user can immediately ask anything.
                briefing = profile.briefing_time or self._default_briefing_time
                await uow.profiles.upsert(user_id, briefing_time=briefing)
                await self._ensure_morning_brief(uow, user_id, briefing)
                await self._complete(uow, user_id)
                return OnboardingReply(completed=True, followed_by_agent=True)
            # The very first message may already answer the role question.
            role = _parse_role(text)
            if role:
                await uow.profiles.upsert(user_id, role=role)
                await uow.memories.upsert_observation(
                    user_id,
                    memory_key="user_profile",
                    value={"role": role},
                    summary=f"user role: {role}",
                    confidence=0.9,
                )
                await self._set_step(uow, user_id, "interests")
                return OnboardingReply(text=_INTERESTS_THEN, still_onboarding=True)
            await self._set_step(uow, user_id, "role")
            return OnboardingReply(text=_WELCOME, still_onboarding=True)

        if step == "role":
            role = _parse_role(text)
            if role:
                await uow.profiles.upsert(user_id, role=role)
                await uow.memories.upsert_observation(
                    user_id,
                    memory_key="user_profile",
                    value={"role": role},
                    summary=f"user role: {role}",
                    confidence=0.9,
                )
            await self._set_step(uow, user_id, "interests")
            return OnboardingReply(text=_INTERESTS_THEN, still_onboarding=True)

        if step == "interests":
            interests = _parse_interests(text)
            if interests:
                await uow.profiles.upsert(user_id, interests=interests)
                await uow.memories.upsert_observation(
                    user_id,
                    memory_key="user_interests",
                    value={"interests": interests},
                    summary=f"interested in: {', '.join(interests)}",
                    confidence=0.85,
                )
            # The monitor question also asks for tickers: capture them here.
            for symbol in _parse_symbols(text):
                if await uow.watchlist.get_by_symbol(user_id, symbol) is None:
                    await uow.watchlist.add(user_id, symbol=symbol, name=None, sector=None)
            await self._set_step(uow, user_id, "briefing")
            return OnboardingReply(text=_WATCHLIST_THEN, still_onboarding=True)

        if step == "briefing":
            time_value, _ = _parse_time(text)
            briefing_time = time_value or profile.briefing_time or self._default_briefing_time
            await uow.profiles.upsert(user_id, briefing_time=briefing_time)
            await self._ensure_morning_brief(uow, user_id, briefing_time)
            # Resilience: tickers arriving with the time answer still land in
            # the watchlist instead of being silently dropped.
            for symbol in _parse_symbols(text):
                if await uow.watchlist.get_by_symbol(user_id, symbol) is None:
                    await uow.watchlist.add(user_id, symbol=symbol, name=None, sector=None)
            await self._set_step(uow, user_id, "reminders")
            return OnboardingReply(text=_BRIEFING_THEN, still_onboarding=True)

        if step == "reminders":
            if self._google_connect_available:
                await self._set_step(uow, user_id, "connect")
                return OnboardingReply(text=_CONNECT_THEN, still_onboarding=True)
            return await self._finish(uow, user_id)

        if step == "connect":
            # Any answer (including 'skip') is fine; the user can also connect
            # later by asking the agent, so the optional step never blocks.
            return await self._finish(uow, user_id)

        # Any unknown/unexpected state (including legacy "watchlist" steps
        # persisted by older builds): finalize.
        return await self._finish(uow, user_id)

    async def _finish(self, uow: UnitOfWork, user_id: uuid.UUID) -> OnboardingReply:
        name = ""
        user = await uow.users.get_by_id(user_id)
        if user is not None and user.first_name:
            name = f", {user.first_name}"
        profile = await uow.profiles.get_by_user_id(user_id)
        briefing_time = (
            profile.briefing_time if profile is not None else None
        ) or self._default_briefing_time
        await self._complete(uow, user_id)
        return OnboardingReply(
            text=_DONE.format(name=name, brief=f"briefing at {briefing_time}"),
            completed=True,
        )

    async def _set_step(self, uow: UnitOfWork, user_id: uuid.UUID, step: str) -> None:
        await uow.profiles.upsert(
            user_id,
            onboarding_status=OnboardingStatus.IN_PROGRESS,
            onboarding_context={"step": step},
        )
        await uow.commit()

    async def _complete(self, uow: UnitOfWork, user_id: uuid.UUID) -> None:
        await uow.profiles.upsert(
            user_id,
            onboarding_status=OnboardingStatus.COMPLETED,
            onboarding_context={"step": "done"},
        )
        await uow.commit()

    async def _has_morning_brief(self, uow: UnitOfWork, user_id: uuid.UUID) -> bool:
        jobs = await uow.jobs.list_enabled()
        return any(j.user_id == user_id and j.job_type == "daily_brief" for j in jobs)

    async def _ensure_morning_brief(
        self, uow: UnitOfWork, user_id: uuid.UUID, briefing_time: str
    ) -> None:
        if await self._has_morning_brief(uow, user_id):
            return
        hour, _, minute = briefing_time.partition(":")
        minute = minute or "00"
        await uow.jobs.create(
            job_type="daily_brief",
            cron_expr=f"{int(minute)} {int(hour)} * * *",
            user_id=user_id,
            params={"scope": "both"},
            timezone="UTC",
        )


# --- parsing helpers ----------------------------------------------------------


def _is_skip(text: str | None) -> bool:
    if text is None:
        return False
    t = text.strip().lower()
    return t in _SKIP or (t.split()[0] if t else "") in _SKIP


def _wants_agent(text: str | None) -> bool:
    if text is None:
        return False
    if "?" in text:
        return True
    t = text.strip().lower().strip(".,!?")
    return t in {
        "help",
        "help me",
        "who are you",
        "what are you",
        "what can you do",
        "what do you do",
        "how can you help",
        "tell me about yourself",
        "start over",
        "restart",
        "who made you",
        "are you a bot",
    }


def _parse_role(text: str | None) -> str | None:
    if text is None:
        return None
    t = text.strip().strip(".,!?")
    if not t or _is_skip(text) or t.lower() in _GREETINGS:
        return None
    # Strip conversational prefixes: "i'm an investor"/"i am a founder".
    candidate = re.sub(
        r"(?i)^(?:i'?m|i am|i am an?|my role is|my job is|as an?)\s+",
        "",
        t,
        count=1,
    ).strip()
    candidate = re.sub(r"^(?:a|an|the)\s+", "", candidate).strip()
    if "," in candidate:
        candidate = candidate.split(",")[0].strip()
    candidate = " ".join(candidate.split())
    if not candidate or len(candidate) > 40 or candidate.lower() in _SKIP:
        return None
    if candidate.lower() in _ROLES or len(candidate.split()) <= 3:
        return candidate.title() if candidate.islower() else candidate
    return None


def _parse_interests(text: str | None) -> list[str]:
    if text is None or _is_skip(text):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for part in re.split(r"[,;/\n]", text):
        token = part.strip().lower().rstrip(".")
        for kw in sorted(_KNOWN_INTERESTS, key=len, reverse=True):
            if kw in token:
                if kw not in seen:
                    seen.add(kw)
                    out.append(kw)
                token = token.replace(kw, "")
        # leftover single tokens that look like topics (not short/stop words)
        leftover = [w for w in re.findall(r"[a-z]{4,}", token) if w not in _STOP]
        for w in leftover:
            if w not in seen:
                seen.add(w)
                out.append(w)
            if len(out) >= 8:
                return out
    return out[:8]


def _parse_symbols(text: str | None) -> list[str]:
    if text is None or _is_skip(text):
        return []
    symbols: list[str] = []
    seen: set[str] = set()
    for match in _SYMBOL_RE.findall(text):
        symbol = match.upper()
        if symbol in _STOP or not re.match(r"^[A-Z][A-Z0-9]*$", symbol):
            continue
        if symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
        if len(symbols) >= 10:
            break
    return symbols


def _parse_time(text: str | None) -> tuple[str | None, bool]:
    if text is None or _is_skip(text):
        return None, False
    m = _TIME_RE.search(text.strip().lower())
    if not m:
        return None, False
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    period = (m.group(3) or "").replace(".", "")[:2]
    if period == "pm":
        hour = hour + 12 if hour < 12 else hour
    elif period == "am":
        hour = 0 if hour == 12 else hour
    elif hour > 23:
        return None, False
    return f"{hour:02d}:{minute:02d}", True


__all__ = ["OnboardingEngine", "OnboardingReply"]
