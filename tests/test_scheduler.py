"""Scheduler worker + cron helper tests."""

import datetime as dt

from sqlalchemy import select

from app.application.scheduling.cron import (
    compute_next_run,
    cron_from_local_time,
    extract_clock_time,
    is_valid_cron,
)
from app.application.scheduling.worker import JobRunner, SchedulerWorker


def test_is_valid_cron():
    assert is_valid_cron("0 8 * * *")
    assert is_valid_cron("*/15 * * * *")
    assert not is_valid_cron("")
    assert not is_valid_cron("nope")
    assert not is_valid_cron("60 0 * * *")


def test_compute_next_run_chronological():
    base = dt.datetime(2026, 8, 7, 7, 0, tzinfo=dt.UTC)
    nxt = compute_next_run("30 8 * * *", after=base)
    assert nxt is not None
    assert nxt.hour == 8 and nxt.minute == 30
    later = compute_next_run("30 8 * * *", after=nxt)
    assert later is not None
    assert later > nxt


def test_compute_next_run_none_for_bad_expr():
    assert compute_next_run("oops") is None


def test_cron_from_local_time_utc_identity():
    assert cron_from_local_time("08:30", None) == "30 8 * * *"


def test_cron_from_local_time_offset():
    # 08:30 in Asia/Kolkata = 03:00 UTC that day
    assert cron_from_local_time("08:30", "Asia/Kolkata") == "0 3 * * *"


def test_extract_clock_time():
    assert extract_clock_time("remind me at 9:15am to review") == "09:15"
    assert extract_clock_time("18:30") == "18:30"
    assert extract_clock_time("no time here") is None


async def test_scheduler_sweeps_due_jobs_once(session_factory, db_engine):
    from app.domain.entities import ScheduledJob
    from app.infrastructure.db.session import async_sessionmaker

    def fake_now():
        return dt.datetime(2026, 8, 7, 12, 1, 0, tzinfo=dt.UTC)

    maker = async_sessionmaker(bind=db_engine)
    async with maker() as session:
        session.add(
            ScheduledJob(
                job_type="reminder",
                cron_expr="* * * * *",
                params={"text": "hi"},
                next_run_at=fake_now() - dt.timedelta(minutes=1),
            )
        )
        await session.commit()

    called: list[str] = []

    async def handler(inner_uow, job, context):
        called.append(job.job_type)

    runner = JobRunner({"reminder": handler})

    worker = SchedulerWorker(maker, runner, now_fn=fake_now)
    assert await worker.sweep_once() == 1
    assert called == ["reminder"]

    # second sweep must not re-fire the same occurrence (idempotent run_key)
    assert await worker.sweep_once() == 0


async def test_scheduler_runs_never_scheduled_job_immediately(session_factory, db_engine):
    """Jobs created without next_run_at (NULL, like ensure_cycle_jobs) must fire
    on the next sweep instead of being skipped forever."""
    from app.domain.entities import ScheduledJob
    from app.infrastructure.db.session import async_sessionmaker

    def fake_now():
        return dt.datetime(2026, 8, 7, 12, 1, 0, tzinfo=dt.UTC)

    maker = async_sessionmaker(bind=db_engine)
    async with maker() as session:
        session.add(ScheduledJob(job_type="price_alerts", cron_expr="*/15 * * * *"))
        await session.commit()

    called: list[str] = []

    async def handler(inner_uow, job, context):
        called.append(job.job_type)

    worker = SchedulerWorker(maker, JobRunner({"price_alerts": handler}), now_fn=fake_now)
    assert await worker.sweep_once() == 1
    assert called == ["price_alerts"]

    # after the immediate run the job gets a real next_run_at on the boundary
    async with maker() as session:
        job = (
            await session.execute(
                select(ScheduledJob).where(ScheduledJob.job_type == "price_alerts")
            )
        ).scalar_one()
        assert job.last_run_at is not None
        assert job.next_run_at is not None
        assert job.next_run_at.replace(tzinfo=dt.UTC) > fake_now()

    # already covered occurrence does not re-fire
    assert await worker.sweep_once() == 0


async def test_misfired_job_recovers_to_next_boundary(session_factory, db_engine):
    """A job that missed its window (e.g. instance was asleep) must advance to
    its next fire time instead of staying permanently misfired."""
    from app.domain.entities import ScheduledJob
    from app.infrastructure.db.session import async_sessionmaker

    def fake_now():
        return dt.datetime(2026, 8, 7, 3, 30, 0, tzinfo=dt.UTC)

    maker = async_sessionmaker(bind=db_engine)
    async with maker() as session:
        session.add(
            ScheduledJob(
                job_type="news_alerts",
                cron_expr="*/30 * * * *",
                next_run_at=fake_now() - dt.timedelta(minutes=40),
            )
        )
        await session.commit()

    worker = SchedulerWorker(maker, JobRunner({}), now_fn=fake_now)
    assert await worker.sweep_once() == 0

    async with maker() as session:
        job = (
            await session.execute(
                select(ScheduledJob).where(ScheduledJob.job_type == "news_alerts")
            )
        ).scalar_one()
        assert job.next_run_at is not None
        assert job.next_run_at.replace(tzinfo=dt.UTC) > fake_now()


def test_runner_has_types():
    runner = JobRunner({"a": lambda: None})
    assert runner.has("a")
    assert not runner.has("missing")
