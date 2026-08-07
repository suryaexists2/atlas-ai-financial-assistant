"""One-off reminder handler for the scheduler worker."""

from __future__ import annotations

from app.application.intelligence import IntelligenceContext, enqueue_for_user
from app.infrastructure.db.uow import UnitOfWork


async def fire_reminder(uow: UnitOfWork, job, ctx: IntelligenceContext) -> None:
    text = (job.params or {}).get("text")
    if not text:
        return
    await enqueue_for_user(uow, job, f"⏰ Reminder: {text}", priority=9)
    if (job.params or {}).get("once"):
        # One-off reminder: retire the job so it never fires again.
        # The worker's sweep commits this at the end of the cycle.
        job.enabled = False


__all__ = ["fire_reminder"]
