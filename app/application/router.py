"""Deterministic intent routing.

Classifies an incoming message without any LLM turn: cheap keyword patterns
decide whether the reply is a canned template (greeting, identity, scope
refusal, confirmation) or an agent turn with a scoped tool group (market,
watchlist, alerts, reminders, documents, google). Anything ambiguous falls
through to `complex`, which runs the full agent with all tools.

Routing is pure: no I/O, no randomness, no model calls — fully testable.
"""

from __future__ import annotations

import re

# --- Intents ---------------------------------------------------------------
GREETING = "greeting"
IDENTITY = "identity"
CONFIRM = "confirm"
SCOPE = "scope"
MARKET = "market"
WATCHLIST = "watchlist"
ALERTS = "alerts"
REMINDERS = "reminders"
DOCUMENTS = "documents"
GOOGLE = "google"
COMPLEX = "complex"

# Intents answered with a fixed template: zero LLM tokens, zero tools.
TEMPLATE_INTENTS = frozenset({GREETING, IDENTITY, CONFIRM, SCOPE})

# Agent intents with a restricted tool group + trimmed context.
SCOPED_INTENTS = frozenset({MARKET, WATCHLIST, ALERTS, REMINDERS, DOCUMENTS, GOOGLE})

# --- Greeting --------------------------------------------------------------
_GREETING_RE = re.compile(
    r"(?i)^\s*(?:hi|hello|hey|yo|hola|namaste|namaskar|salaam)"
    r"(?:[,!. ]+(?:guys|there|atlas|bot)?)?[?!.\s]*$"
    r"|^\s*good\s*(?:morning|afternoon|evening)\s*[!.?\s]*$"
)

# --- Scope refusal ---------------------------------------------------------
_SCOPE_RE = re.compile(
    r"(?i)\b(?:"
    r"joke|jokes|poem|poems|story|stories|essay|essays|song|songs|lyrics|"
    r"recipe|recipes|cookbook|baking|bake|cook|"
    r"code|python|javascript|html|css|sql|program|function|coding|script|debug|"
    r"translate|translation|"
    r"math|maths|solve|quiz|trivia|riddle|homework|terrible|"
    r"sing|dance|draw|paint|portrait|photo\s*edit|"
    r"write\s+me(?:$|\s+(?:a|an|the|some|my))|compose\s+(?:a|me)|"
    r"weather|temperature|forecast\b(?!\s*(?:market|stock|earning))|"
    r"movie|movies|game|games|book\s+recommendation|"
    r"ticket|booking|hotel|flight|uber|ola|food|restaurant"
    r")(?:\b|$)"
)


# --- Task keywords ----------------------------------------------------------
_GOOGLE_RE = re.compile(
    r"(?i)\b(?:gmail|email|emails|mail|calendar|drive|google\s*(?:sheet|sheets|"
    r"doc|docs|account)|sheets?|spreadsheet|connect\s+google)\b"
)

_WATCHLIST_RE = re.compile(
    r"(?i)\b(?:watchlist|watch\s+list|watching|tracked|tracking|track\b|watch\b)\b"
)

_ALERTS_RE = re.compile(
    r"(?i)\b(?:alert|alerts|alert\s+me|price\s+alert|news\s+alert|"
    r"filing\s+alert|notify|notification|notifications)\b"
)

_REMINDER_RE = re.compile(
    r"(?i)\b(?:remind|reminder|reminders|schedule|appointment|appointments|"
    r"briefing|daily\s+brief|meeting\s*(?:today|tomorrow)?)\b"
    r"|\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b"
)

_DOCUMENT_RE = re.compile(
    r"(?i)\b(?:document|documents|pdf|pdfs|xlsx|docx|csv|json|attachment|"
    r"attachments|file\b|files\b|my\s+report|the\s+document|that\s+file|"
    r"last\s+report|read\s+that)\b"
)

_MARKET_RE = re.compile(
    r"(?i)\b(?:quote|quotes|price|prices|stock|stocks|market|markets|trading|"
    r"ticker|earnings|filing|filings|ipo|dividend|eps|index|indices|nifty|"
    r"sensex|s&p\s*500|nasdaq|dow\s+jones|"
    r"aapl|msft|googl|amzn|nvda|tsla|meta|brk[.-]?b)\b"
    r"|\$\s?[A-Z]{1,5}\b"
)


def classify(text: str | None, *, is_media: bool = False) -> str:
    """One deterministic pass over the user's text to pick an intent.

    Order matters: fixed replies beat scoped groups (a word like 'alert' must
    not be swallowed by a market keyword), and scope refusals beat everything
    so a single off-topic keyword can't fake a financial turn. Messages with
    no text or only a media attachment route to `documents`, and anything
    ambiguous falls to `complex`.
    """
    if text is None or not text.strip():
        return DOCUMENTS if is_media else COMPLEX
    t = text.strip()
    if _GREETING_RE.match(t):
        return GREETING
    if _SCOPE_RE.search(t):
        return SCOPE
    if _acks(t):
        return CONFIRM
    if _is_identity(t):
        return IDENTITY
    if _GOOGLE_RE.search(t):
        return GOOGLE
    if _WATCHLIST_RE.search(t):
        return WATCHLIST
    if _ALERTS_RE.search(t):
        return ALERTS
    if _REMINDER_RE.search(t):
        return REMINDERS
    if _DOCUMENT_RE.search(t):
        return DOCUMENTS
    if _MARKET_RE.search(t):
        return MARKET
    return COMPLEX


def _acks(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)^\s*(?:thanks|thank\s+you|ty|thx)|\b(?:ok|okay|sure|kk|"
            r"got\s+it|understood|nice|great|awesome|perfect|cool|"
            r"sounds\s+good|fine|works?)\b[!.?]*\s*$",
            text,
        )
    )


def _is_identity(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)\b(?:who|what)\s+(?:are|is)\s+(?:you|atlas)\b"
            r"|\bwhat\s+(?:can|do)\s+you\s+(?:do|help)\b"
            r"|\bwhat(?:'s|\sis)?\s+your\s+(?:purpose|role)\b"
            r"|\btell\s+me\s+about\s+yourself\b"
            r"|\b(?:are|r)\s+you\s+(?:a\s+)?(?:bot|ai|robot|chatgpt)\b"
            r"|^\s*help\s*[!.?]*\s*$",
            text,
        )
    )


def template_reply(intent: str) -> str | None:
    """Fixed replies for template intents; None for agent-scoped intents."""
    if intent == GREETING:
        return (
            "Hi! I'm Atlas, your AI financial assistant. "
            "Ask me about a stock price, market news, company filings, your "
            "watchlist, alerts, reminders, or meetings — or say 'help' to see "
            "what I can do."
        )
    if intent == CONFIRM:
        return (
            "You're welcome! Whenever you're ready, I can help with live "
            "quotes, market and news analysis, SEC filings, your watchlist, "
            "alerts, reminders, or meetings. What would you like?"
        )
    if intent == IDENTITY:
        return (
            "I'm Atlas, your AI financial assistant. "
            "I can help you with live quotes, market and news analysis, company "
            "research, SEC filings, reports and documents, watchlists, alerts, "
            "reminders, and meetings — plus your connected Gmail, Calendar, Drive, "
            "and Sheets. What would you like to look into?"
        )
    if intent == SCOPE:
        return (
            "I'm Atlas, a financial assistant — so I focus on markets, quotes, "
            "news, filings, documents, watchlists, alerts, reminders, and "
            "meetings. That one's outside my lane, but I'd love to help with "
            "something financial."
        )
    return None


__all__ = [
    "GREETING", "IDENTITY", "CONFIRM", "SCOPE", "MARKET", "WATCHLIST",
    "ALERTS", "REMINDERS", "DOCUMENTS", "GOOGLE", "COMPLEX",
    "TEMPLATE_INTENTS", "SCOPED_INTENTS", "classify", "template_reply",
]
