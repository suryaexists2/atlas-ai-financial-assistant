"""Idempotent seed script for local development.

Creates a demo user with a profile, watchlist, memory entries, a morning-brief
scheduled job and a couple of outbound test messages. Safe to run repeatedly.
"""

import asyncio
import datetime as dt

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.domain.enums import AlertKind, OnboardingStatus
from app.infrastructure.db.session import build_engine, build_session_factory, init_db
from app.infrastructure.db.uow import UnitOfWork

logger = get_logger("seed")

DEMO_TELEGRAM_ID = 123456789


async def seed() -> None:
    settings = get_settings()
    configure_logging("INFO")
    engine = build_engine(settings)
    await init_db(engine)

    uow = UnitOfWork(build_session_factory(engine))
    async with uow:
        existing = await uow.users.get_by_telegram_id(DEMO_TELEGRAM_ID)
        if existing is None:
            user = await uow.users.create(
                telegram_id=DEMO_TELEGRAM_ID,
                username="demo_investor",
                first_name="Demo",
                last_name="Investor",
                timezone="America/New_York",
            )
            await uow.profiles.upsert(
                user.id,
                role="Investor",
                interests=["Semiconductors", "AI", "Big Tech"],
                briefing_enabled=True,
                briefing_time="08:00",
                onboarding_status=OnboardingStatus.COMPLETED,
            )
            for symbol, name, sector in [
                ("NVDA", "NVIDIA", "Semiconductors"),
                ("MSFT", "Microsoft", "Software"),
                ("GOOG", "Alphabet", "Software"),
                ("TSLA", "Tesla", "Automotive"),
            ]:
                await uow.watchlist.add(user.id, symbol=symbol, name=name, sector=sector)

            await uow.alerts.create(
                user.id,
                kind=AlertKind.PRICE,
                symbol="NVDA",
                condition={"op": "gte", "threshold_pct": 5},
            )

            await uow.memories.upsert_observation(
                user.id,
                memory_key="interest:ai",
                value={"label": "AI / Semiconductors"},
                summary="Actively follows AI and semiconductor sector",
                confidence=0.9,
            )

            now = dt.datetime.now(dt.UTC).replace(second=0, microsecond=0)
            await uow.jobs.create(
                job_type="morning_brief",
                cron_expr="0 8 * * *",
                user_id=user.id,
                params={"timezone": "America/New_York"},
                timezone="America/New_York",
            )
            await uow.jobs.create(
                job_type="watchlist_alert_check",
                cron_expr="*/5 * * * *",
                user_id=user.id,
                params={"max_pct_move_per_session": 5},
            )
            logger.info("seeded_demo_user", telegram_id=DEMO_TELEGRAM_ID, at=now.isoformat())
        else:
            logger.warning("demo_user_already_exists", telegram_id=DEMO_TELEGRAM_ID)
        await uow.commit()


if __name__ == "__main__":
    asyncio.run(seed())
