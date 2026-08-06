"""Reply composers: dev echo and the real agent turn.

The dev echo is used in echo mode; the AgentComposer drives the M3 AgentCore
from a ReplyContext, wiring the injected providers into the tool context.
"""

from __future__ import annotations

from app.application.agent.core import AgentCore
from app.application.agent.tools import ToolContext
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
    def __init__(
        self,
        agent: AgentCore,
        *,
        finnhub: FinnhubClient | None = None,
        sec: SecEdgarClient | None = None,
    ) -> None:
        self._agent = agent
        self._finnhub = finnhub
        self._sec = sec

    async def __call__(self, ctx: ReplyContext) -> str | None:
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
