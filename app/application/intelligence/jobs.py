"""Seed the global alert-monitoring jobs once (idempotent)."""

from __future__ import annotations

from app.infrastructure.db.uow import UnitOfWork

_CYCLE_JOBS: dict[str, str] = {
    "price_alerts": "*/15 * * * *",  # re-checks price thresholds frequently
    "news_alerts": "*/30 * * * *",  # company news is slower-moving
    "filing_alerts": "*/30 * * * *",  # SEC filings land throughout the day
}


async def ensure_cycle_jobs(uow: UnitOfWork) -> None:
    """Ensures the three global monitoring jobs exist (no user attached)."""
    existing = {j.job_type for j in await uow.jobs.list_enabled()}
    for job_type, cron_expr in _CYCLE_JOBS.items():
        if job_type in existing:
            continue
        await uow.jobs.create(
            job_type=job_type,
            cron_expr=cron_expr,
            user_id=None,
            params={"scope": "all_users"},
            timezone="UTC",
        )


__all__ = ["ensure_cycle_jobs"]
