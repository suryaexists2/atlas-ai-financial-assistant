"""AgentCore: the M3 turn loop.

Runs one agent turn: builds context, calls the LLM gateway, executes any
requested tool calls, and loops until the model produces a final text reply
(or the round budget is exhausted). The loop is pure orchestration — all
state changes happen through the injected uow/repos and provider clients.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from app.application.agent.context import build_messages
from app.application.agent.ports import LLMGateway
from app.application.agent.tools import ToolContext, ToolRegistry
from app.core.logging import get_logger
from app.infrastructure.db.uow import UnitOfWork
from app.infrastructure.llm.gateway import LLMGatewayError

logger = get_logger(__name__)


class AgentCore:
    def __init__(
        self,
        gateway: LLMGateway,
        tools: ToolRegistry,
        *,
        max_tool_rounds: int = 5,
        max_tokens: int = 600,
        temperature: float = 0.3,
        fallback_reply: str = "Sorry — I hit a temporary hiccup. Try again in a moment.",
        max_context_messages: int = 24,
        debug_reply_errors: bool = False,
    ) -> None:
        self._gateway = gateway
        self._tools = tools
        self._max_tool_rounds = max_tool_rounds
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._fallback_reply = fallback_reply
        self._max_context_messages = max_context_messages
        self._debug_reply_errors = debug_reply_errors
        self.last_error: str | None = None

    def _degrade(self, reason: str) -> str:
        """Records why the turn fell back and returns the reply (optionally with
        the real error appended while debugging)."""
        self.last_error = reason
        logger.warning("agent_fallback_triggered", reason=reason)
        if self._debug_reply_errors:
            return f"{self._fallback_reply}\n\n[debug {reason[:400]}]"
        return self._fallback_reply

    async def run(
        self,
        uow: UnitOfWork,
        *,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        tool_context: ToolContext | None = None,
    ) -> str:
        """Runs a turn and always returns a reply string (fallback on failure)."""
        self.last_error = None
        started = asyncio.get_running_loop().time()
        messages = await build_messages(
            uow,
            user_id=user_id,
            conversation_id=conversation_id,
            max_messages=self._max_context_messages,
        )
        logger.info(
            "agent_context_built",
            ms=round((asyncio.get_running_loop().time() - started) * 1000),
            n_messages=len(messages),
        )
        tool_ctx = tool_context or ToolContext(uow=uow, user_id=user_id)

        try:
            for _ in range(self._max_tool_rounds + 1):
                try:
                    response = await self._gateway.complete(
                        messages,
                        tools=self._tools.schemas(),
                        max_tokens=self._max_tokens,
                        temperature=self._temperature,
                    )
                except LLMGatewayError as exc:
                    logger.warning("agent_gateway_error", error=str(exc))
                    return self._degrade(f"gateway_error: {exc}")

                if not response.tool_calls:
                    text = (response.content or "").strip()
                    if text:
                        return text
                    logger.warning("agent_empty_reply_falling_back")
                    return self._degrade("empty_reply content  neither text nor tool_calls")

                # Assistant message carrying the tool calls, then results.
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in response.tool_calls
                    ],
                }
                messages.append(assistant_msg)

                for call in response.tool_calls:
                    result = await self._tools.execute(tool_ctx, call.name, call.arguments)
                    logger.info(
                        "agent_tool_executed",
                        tool=call.name,
                        result_preview=result[:200],
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result,
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - never leave the user without a reply
            logger.exception("agent_turn_error_falling_back")
            return self._degrade(f"turn_exception: {type(exc).__name__}: {exc}")

        logger.warning("agent_tool_rounds_exhausted", rounds=self._max_tool_rounds)
        return self._degrade("tool_rounds_exhausted")


__all__ = ["AgentCore"]
