"""Conversation/data reset service: /reset command, two-step confirmation,
and a full per-user wipe.

The reset is deterministic and never touches the LLM: the composer guards
`/reset` before onboarding/agent logic. Pending confirmations live in the
profile's `onboarding_context` JSON (no schema change needed) and expire after
a short TTL so a stale "yes" can never wipe a chat hours later.
"""

from __future__ import annotations

import datetime as dt
import re
import uuid
from dataclasses import dataclass

from app.domain.enums import OnboardingStatus

_PENDING_KEY = "reset_pending_at"
_RESET_PENDING_TTL = dt.timedelta(minutes=10)

_RESET_PROMPT = (
    "Are you sure you want to reset? This will clear all your saved data — "
    "chat history, memories, watchlist, alerts, documents, and Google "
    "connections. Reply \"yes\" to confirm, or anything else to cancel."
)
_RESET_ALREADY_PROMPT = (
    "You already have a reset pending. Reply \"yes\" to confirm, or anything "
    "else to cancel."
)
_RESET_CANCELLED = (
    "Reset cancelled — everything is still in place. Ask me anything about "
    "the markets, or send /reset when you really want a fresh start."
)
_RESET_DONE = (
    "All done — your data has been cleared. I'm Atlas, your AI financial "
    "assistant. What would you like to look into today?"
)

_RESET_COMMAND_RE = re.compile(r"^\s*/reset\s*[.!?]*\s*$", re.IGNORECASE)
_RESET_CONFIRM_RE = re.compile(
    r"^\s*(?:y|yes|yeah|yep|yup|sure|ok|okay|confirm|haan|hahn|han|ha)\s*[.!]*\s*$",
    re.IGNORECASE,
)


@dataclass
class ResetTurn:
    """Outcome of one turn for the reset guard.

    `reply` None means the turn is not a reset interaction at all. When
    `wiped` is True the composer must open a fresh conversation and persist
    the reply there (the old conversations are gone).
    """

    reply: str | None = None
    wiped: bool = False


def is_reset_command(text: str | None) -> bool:
    return bool(text and _RESET_COMMAND_RE.match(text))


def is_confirmation(text: str | None) -> bool:
    return bool(text and _RESET_CONFIRM_RE.match(text))


def _pending_at(profile) -> dt.datetime | None:
    raw = (profile.onboarding_context or {}).get(_PENDING_KEY)
    if not raw:
        return None
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed


async def reset_turn(
    uow,
    *,
    user_id: uuid.UUID,
    text: str | None,
    now: dt.datetime | None = None,
) -> ResetTurn:
    """One turn of the reset state machine.

    - `/reset` with no pending confirmation -> prompt + arm the flag.
    - `/reset` with a pending confirmation -> re-prompt.
    - A confirmation while armed -> full wipe, fresh state.
    - Any other message while armed -> cancel and clear the flag.
    - Everything else -> normal flow (reply None).
    """
    now = now or dt.datetime.now(dt.UTC)
    profile = await uow.profiles.get_by_user_id(user_id)
    pending = _pending_at(profile)
    active = pending is not None and (now - pending) <= _RESET_PENDING_TTL

    if is_reset_command(text):
        if active:
            return ResetTurn(reply=_RESET_ALREADY_PROMPT)
        if pending is not None:
            await _clear_pending(uow, user_id, profile)
        context = dict(profile.onboarding_context or {}) if profile else {}
        context[_PENDING_KEY] = now.isoformat()
        await uow.profiles.upsert(user_id, onboarding_context=context)
        return ResetTurn(reply=_RESET_PROMPT)

    if active:
        if is_confirmation(text):
            await _clear_pending(uow, user_id, profile)
            await wipe_user(uow, user_id)
            return ResetTurn(reply=_RESET_DONE, wiped=True)
        await _clear_pending(uow, user_id, profile)
        return ResetTurn(reply=_RESET_CANCELLED)

    if pending is not None:
        await _clear_pending(uow, user_id, profile)
    return ResetTurn()


async def _clear_pending(uow, user_id: uuid.UUID, profile) -> None:
    if profile is None:
        return
    context = dict(profile.onboarding_context or {})
    if _PENDING_KEY in context:
        context.pop(_PENDING_KEY)
        await uow.profiles.upsert(user_id, onboarding_context=context)


async def wipe_user(uow, user_id: uuid.UUID) -> None:
    """Deletes every piece of user data and resets the profile so the user
    starts from a genuinely fresh onboarding state. The users row itself is
    kept (telegram_id linkage); operator logs (outbox, ingest ledger, job
    events) are left untouched."""
    await uow.conversations.delete_for_user(user_id)
    await uow.memories.delete_all_for_user(user_id)
    await uow.watchlist.delete_all_for_user(user_id)
    await uow.alerts.delete_for_user(user_id)
    await uow.jobs.delete_for_user(user_id)
    await uow.documents.delete_for_user(user_id)
    await uow.integrations.delete_all_for_user(user_id)
    await uow.oauth_flows.delete_all_for_user(user_id)
    await uow.profiles.upsert(
        user_id,
        role=None,
        interests=None,
        briefing_enabled=True,
        briefing_time=None,
        onboarding_status=OnboardingStatus.NOT_STARTED,
        onboarding_context={},
    )


__all__ = [
    "ResetTurn",
    "is_reset_command",
    "is_confirmation",
    "reset_turn",
    "wipe_user",
]
