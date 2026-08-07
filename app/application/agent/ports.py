"""Ports the Agent Core depends on (LLM gateway, data providers).

Keeping these as structural protocols means the agent logic is testable with
fakes and swappable without touching the turn loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    # Raw assistant-message keys (truncated) for diagnostics; never sent to the user.
    raw: dict[str, Any] = field(default_factory=dict)


class LLMGateway(Protocol):
    """Chat-completion gateway with tool calling (OpenAI-compatible shape)."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 600,
        temperature: float = 0.3,
    ) -> LLMResponse: ...
