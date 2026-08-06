"""Proactive intelligence jobs: daily briefings, alerts, and reminders.

These handlers run inside the scheduler worker and always deliver through the
durable Telegram outbox. They rely on injected providers (Finnhub/SEC) and an
optional LLM gateway for the natural-language layer; when no LLM is configured
or a provider fails, they degrade to a concise deterministic summary instead
of failing silently.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.application.agent.ports import LLMGateway
from app.infrastructure.db.uow import UnitOfWork
from app.infrastructure.providers.finnhub import FinnhubClient
from app.infrastructure.providers.sec import SecEdgarClient


@dataclass
class IntelligenceContext:
    """Everything a scheduled intelligence job may need to do its work."""

    finnhub: FinnhubClient | None = None
    sec: SecEdgarClient | None = None
    gateway: LLMGateway | None = None
    timezone: str = "UTC"

    def __post_init__(self) -> None:
        self.services: set[str] = set()
        if self.finnhub is not None:
            self.services.add("market")
        if self.sec is not None:
            self.services.add("filings")
        if self.gateway is not None:
            self.services.add("llm")


async def enqueue_for_user(uow: UnitOfWork, job, text: str, *, priority: int = 5) -> bool:
    """Persists one outbound Telegram message for the job's user. False when the
    user or their chat is unknown."""
    if job.user_id is None:
        return False
    user = await uow.users.get_by_id(job.user_id)
    if user is None:
        return False
    await uow.outbox.enqueue(
        chat_id=user.telegram_id,
        payload={
            "type": "text",
            "text": text,
            "correlation_id": f"job:{job.job_type}:{uuid.uuid4().hex[:8]}",
        },
        priority=priority,
    )
    return True


async def chat_id_for_user(uow: UnitOfWork, user_id: uuid.UUID) -> int | None:
    user = await uow.users.get_by_id(user_id)
    return user.telegram_id if user is not None else None


__all__ = ["IntelligenceContext", "chat_id_for_user", "enqueue_for_user"]
