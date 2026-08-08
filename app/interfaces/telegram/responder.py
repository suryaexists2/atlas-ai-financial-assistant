"""Reply composers: dev echo, conversational onboarding, and the agent turn."""

from __future__ import annotations

import re
from typing import Any

from app.application.agent.core import AgentCore
from app.application.agent.tools import ToolContext
from app.application.onboarding import OnboardingEngine, OnboardingReply
from app.application.reset import reset_turn
from app.domain.enums import MessageRole
from app.infrastructure.providers.finnhub import FinnhubClient
from app.infrastructure.providers.sec import SecEdgarClient
from app.interfaces.telegram.normalized import NormalizedMessage
from app.interfaces.telegram.processor import ReplyContext

# Requests to exfiltrate the system prompt / internal instructions. Narrow on
# purpose: it only intercepts meta-attacks; normal questions keep flowing to
# the agent (which also refuses via its rules).
_EXFILTRATION_RE = re.compile(
    r"(?i)\b(?:print|show|repeat|paste|send|reveal|share|disclose|leak|give|"
    r"copy)\s+(?:me\s+|us\s+)?(?:your\s+|the\s+)?(?:full\s+|exact\s+|entire\s+|"
    r"verbatim\s+)?(?:system\s+|developer\s+|base\s+|hidden\s+)?"
    r"(?:prompt|instructions|system prompt|prompt\s+verbatim)\b"
    r"|\bignore\s+(?:all|any)?\s*(?:previous|prior)\s+instructions\b"
    r"|\b(?:system|developer)\s+prompt\s+verbatim\b"
    r"|\bwhat\s+(?:is|are)\s+(?:your|the)\s+(?:system\s+)?"
    r"(?:(?:prompt|instructions)\s+)*(?:prompt|instructions)"
    r"\s+(?:verbatim|exactly|in\s+full)\b"
)

_REFUSAL_REPLY = (
    "I can't share my internal instructions — they're confidential. "
    "Happy to help with what I do best instead: quotes, news, filings, your "
    "documents, reminders, or meetings. What would you like?"
)

# "Who/what are you" and purpose questions get one consistent, deterministic
# answer — Atlas's identity must never depend on which model answers.
_IDENTITY_RE = re.compile(
    r"(?i)\b(?:who|what)\s+(?:are|is)\s+(?:you|atlas)\b"
    r"|\bwhat\s+(?:can|do)\s+you\s+(?:do|help)\b"
    r"|\bwhat(?:'s|\sis)?\s+your\s+(?:purpose|role)\b"
    r"|\btell\s+me\s+about\s+yourself\b"
    r"|\b(?:are|r)\s+you\s+(?:a\s+)?(?:bot|ai|robot|chatgpt)\b"
)

_IDENTITY_REPLY = (
    "I'm Atlas, your AI financial assistant. "
    "I can help you with live quotes, market and news analysis, company "
    "research, SEC filings, reports and documents, watchlists, alerts, "
    "reminders, and meetings — plus your connected Gmail, Calendar, Drive, "
    "and Sheets. What would you like to look into?"
)


class EchoComposer:
    """Dev-only: mirrors back what the user sent."""

    async def __call__(self, ctx: ReplyContext) -> str:
        return dev_echo_reply(ctx.message)


async def dev_echo_reply(message: NormalizedMessage) -> str:
    if message.is_media:
        short_id = message.media_file_id[:12]
        return (
            f"Got it — received your {message.media_type} (id: {short_id}…). "
            "I'll be able to analyze this soon."
        )
    return f"Got it — I heard: {message.combined_text}"


def exfiltration_reply(text: str | None) -> str | None:
    """Deterministic prompt-exfiltration guard: returns a canned refusal when
    the message tries to reveal the system prompt or override instructions,
    else None so the normal reply path runs."""
    if not text:
        return None
    return _REFUSAL_REPLY if _EXFILTRATION_RE.search(text) else None


def identity_reply(text: str | None) -> str | None:
    """Deterministic identity guard: "who/what are you" questions get the
    canonical Atlas intro without an LLM turn, so the answer never varies
    with the underlying model. Returns None for everything else."""
    if not text:
        return None
    return _IDENTITY_REPLY if _IDENTITY_RE.search(text) else None


class AgentComposer:
    """Runs conversational onboarding until the user is set up, then hands off
    to the agent. Onboarding is invisible once completed."""

    def __init__(
        self,
        agent: AgentCore,
        *,
        finnhub: FinnhubClient | None = None,
        sec: SecEdgarClient | None = None,
        google_sheets: Any = None,
        indices: Any = None,
        google_oauth: Any = None,
        media_pipeline: Any = None,
        public_base_url: str | None = None,
        onboarding: OnboardingEngine | None = None,
    ) -> None:
        self._agent = agent
        self._finnhub = finnhub
        self._sec = sec
        self._google_sheets = google_sheets
        self._indices = indices
        self._google_oauth = google_oauth
        self._media_pipeline = media_pipeline
        self._public_base_url = public_base_url
        self._onboarding = onboarding or OnboardingEngine()

    async def __call__(self, ctx: ReplyContext) -> str | None:
        # /reset is deterministic and must run before onboarding (it has to
        # work in every state, including mid-onboarding). No LLM turn ever.
        reset = await reset_turn(
            ctx.uow,
            user_id=ctx.user_id,
            text=ctx.message.combined_text,
        )
        if reset.wiped:
            # Old conversations are gone; persist this reply in a fresh one so
            # the new chat starts clean (and the processor skips its own copy).
            fresh = await ctx.uow.conversations.create(ctx.user_id)
            await ctx.uow.conversations.add_message(
                fresh.id,
                role=MessageRole.ASSISTANT,
                content=reset.reply,
            )
            ctx.assistant_persisted = True
        if reset.reply is not None:
            return reset.reply

        onboarding_reply: OnboardingReply = await self._onboarding.turn(
            ctx.uow,
            user_id=ctx.user_id,
            text=ctx.message.combined_text,
            is_media=ctx.message.is_media,
        )
        if onboarding_reply.text is not None:
            # Onboarding is active, or just finished with a message — that
            # message IS the reply for this turn.
            return onboarding_reply.text
        if not onboarding_reply.completed:
            return None
        # Prompt-exfiltration attempts are answered deterministically: no LLM
        # turn, no chance of a leak, no wasted tool rounds.
        refusal = exfiltration_reply(ctx.message.combined_text)
        if refusal:
            return refusal
        # Identity questions get one consistent, model-independent answer.
        identity = identity_reply(ctx.message.combined_text)
        if identity:
            if onboarding_reply.notice:
                return f"{onboarding_reply.notice}\n\n{identity}"
            return identity
        # Onboarding already done (or a question exited it): run the agent.
        tool_ctx = ToolContext(
            uow=ctx.uow,
            user_id=ctx.user_id,
            finnhub=self._finnhub,
            sec=self._sec,
            google_sheets=self._google_sheets,
            indices=self._indices,
            google_oauth=self._google_oauth,
            media_pipeline=self._media_pipeline,
            public_base_url=self._public_base_url,
            chat_id=ctx.message.chat_id,
        )
        reply = await self._agent.run(
            ctx.uow,
            user_id=ctx.user_id,
            conversation_id=ctx.conversation_id,
            tool_context=tool_ctx,
        )
        if onboarding_reply.notice and reply:
            # First-time user who skipped onboarding (question-first, media-
            # first, or 'skip'): prepend the one-time testing-mode heads-up to
            # the agent's first reply so they still get the disclaimer.
            reply = f"{onboarding_reply.notice}\n\n{reply}"
        if tool_ctx.oauth_connect_url:
            # Contextual, single-purpose OAuth button — the only inline button
            # the bot ever emits; everything else stays plain text.
            ctx.reply_markup = {
                "inline_keyboard": [
                    [
                        {
                            "text": "Connect Google",
                            "url": tool_ctx.oauth_connect_url,
                        }
                    ]
                ]
            }
        if getattr(self._agent, "last_error", None):
            ctx.note["agent_error"] = self._agent.last_error
            ctx.note["fallback_used"] = True
        if getattr(self._agent, "last_model", None):
            ctx.note["model"] = self._agent.last_model
        return reply


__all__ = ["AgentComposer", "EchoComposer", "dev_echo_reply"]
