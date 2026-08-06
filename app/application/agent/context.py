"""Agent context manager.

Assembles the system prompt and chat history for one agent turn from
persistence: user profile, active memories, watchlist, and the recent
conversation. Keeps the prompt under a simple message budget and never
exposes internal mechanics to the model beyond what the prompt states.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.infrastructure.db.uow import UnitOfWork

SYSTEM_PROMPT = """You are Atlas, a concise, accurate financial assistant in a Telegram chat.

Rules:
- Answer in plain, conversational language. No markdown headers or bullet spam;
  one or two short paragraphs at most.
- Use tools for any market data, company information, or SEC filings.
  Never invent prices or figures.
- If you do not know something or a tool returns no data, say so plainly
  instead of guessing.
- Flag uncertainty and time-sensitivity (e.g. "prices may be delayed").
- Quote figures with context: price, change, and change percent.
- Keep the user's memory in mind (preferences, watchlist, interests) but do
  not mention the memory system itself.
- If the user asks to track or remember something, use the memory tools.
- Be helpful but brief; silence is better than filler."""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


async def build_messages(
    uow: UnitOfWork,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    max_messages: int = 24,
) -> list[dict[str, Any]]:
    """Assembles [system, (context facts), ...recent conversation] for the LLM."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    profile = await uow.profiles.get_by_user_id(user_id)
    if profile is not None and (profile.interests or profile.role):
        parts = []
        if profile.role:
            parts.append(f"role: {profile.role}")
        if profile.interests:
            parts.append(f"interests: {', '.join(profile.interests)}")
        messages.append({"role": "system", "content": "User profile: " + "; ".join(parts)})

    watchlist = await uow.watchlist.list_active(user_id)
    if watchlist:
        symbols = ", ".join(item.symbol for item in watchlist)
        messages.append({"role": "system", "content": f"User watchlist: {symbols}"})

    memories = await uow.memories.list_active(user_id, limit=20)
    if memories:
        lines = [f"- {m.summary}" for m in memories if m.summary]
        if lines:
            messages.append(
                {
                    "role": "system",
                    "content": "Known facts about the user:\n" + "\n".join(lines),
                }
            )

    history = await uow.conversations.list_messages(conversation_id, limit=max_messages)
    for message in history:
        role = message.role.value if message.role else "user"
        if role not in ("user", "assistant"):
            continue
        content = message.content or ""
        if not content.strip():
            if role == "user" and message.content_type is not None:
                label = _media_label(message)
                if label:
                    messages.append({"role": "user", "content": label})
            continue
        messages.append({"role": role, "content": content})

    return messages


def _media_label(message: Any) -> str | None:
    """Placeholder text for a media message with no caption/transcript."""
    from app.domain.enums import ContentType

    kind = message.content_type
    if kind == ContentType.VOICE:
        return "[voice message — audio transcription not available yet]"
    if kind == ContentType.IMAGE:
        return "[image message — visual analysis not available yet]"
    if kind == ContentType.DOCUMENT:
        return "[document message]"
    return None


__all__ = ["SYSTEM_PROMPT", "build_system_prompt", "build_messages"]
