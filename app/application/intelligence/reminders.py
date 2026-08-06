"""One-off reminder handler for the scheduler worker."""

from __future__ import annotations

from app.application.intelligence import IntelligenceContext, enqueue_for_user
from app.infrastructure.db.uow import UnitOfWork


async def fire_reminder(uow: UnitOfWork, job, ctx: IntelligenceContext) -> None:
    text = (job.params or {}).get("text")
    if not text:
        return
    await enqueue_for_user(uow, job, f"⏰ Reminder: {text}", priority=9)


__all__ = ["fire_reminder"]
