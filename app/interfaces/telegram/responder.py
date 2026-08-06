"""Reply composers: dev echo, conversational onboarding, and the agent turn."""

from __future__ import annotations

from app.application.agent.core import AgentCore
from app.application.agent.tools import ToolContext
from app.application.onboarding import OnboardingEngine, OnboardingReply
from app.infrastructure.providers.finnhub import FinnhubClient
from app.infrastructure.providers.sec import SecEdgarClient
from app.interfaces.telegram.normalized import NormalizedMessage
from app.interfaces.telegram.processor import ReplyContext


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


class AgentComposer:
    """Runs conversational onboarding until the user is set up, then hands off
    to the agent. Onboarding is invisible once completed."""

    def __init__(
        self,
        agent: AgentCore,
        *,
        finnhub: FinnhubClient | None = None,
        sec: SecEdgarClient | None = None,
        onboarding: OnboardingEngine | None = None,
    ) -> None:
        self._agent = agent
        self._finnhub = finnhub
        self._sec = sec
        self._onboarding = onboarding or OnboardingEngine()

    async def __call__(self, ctx: ReplyContext) -> str | None:
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
        # Onboarding already done (or a question exited it): run the agent.
        tool_ctx = ToolContext(
            uow=ctx.uow,
            user_id=ctx.user_id,
            finnhub=self._finnhub,
            sec=self._sec,
        )
        return await self._agent.run(
            ctx.uow,
            user_id=ctx.user_id,
            conversation_id=ctx.conversation_id,
            tool_context=tool_ctx,
        )


__all__ = ["AgentComposer", "EchoComposer", "dev_echo_reply"]
