"""Looks up the intelligence handlers into a JobRunner."""

from __future__ import annotations

from app.application.intelligence import IntelligenceContext
from app.application.intelligence.alerts import (
    run_filing_alerts,
    run_news_alerts,
    run_price_alerts,
)
from app.application.intelligence.briefing import daily_brief
from app.application.intelligence.reminders import fire_reminder
from app.application.scheduling.worker import JobRunner


def build_intelligence_runner(ctx: IntelligenceContext) -> JobRunner:
    """Maps scheduled `job_type` values to their handlers."""
    return JobRunner(
        {
            "daily_brief": daily_brief,
            "price_alerts": run_price_alerts,
            "news_alerts": run_news_alerts,
            "filing_alerts": run_filing_alerts,
            "reminder": fire_reminder,
        }
    )


__all__ = ["build_intelligence_runner"]
