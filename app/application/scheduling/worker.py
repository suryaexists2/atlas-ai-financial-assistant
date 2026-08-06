"""DB-backed scheduler worker.

Every due job is: locked, ledgered (idempotent via JobEvent unique key), and
dispatched to a registered handler. Handlers enqueue Telegram messages through
the durable outbox — the scheduler never talks to Telegram itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
from typing import Any, Awaitable, Callable, Protocol

from app.application.scheduling.cron import UTC, compute_next_run
from app.infrastructure.db.session import async_sessionmaker
from app.infrastructure.db.uow import UnitOfWork

logger = logging.getLogger(__name__)


class JobHandler(Protocol):
    async def __call__(
        self, uow: UnitOfWork, job: Any, context: Any
    ) -> None: ...


class JobRunner:
    """Resolves a scheduled job's job_type to a handler callable."""

    def __init__(self, handlers: dict[str, Callable[..., Awaitable[None]]]) -> None:
        self._handlers = dict(handlers)

    def register(self, job_type: str, handler: Callable[..., Awaitable[None]]) -> None:
        self._handlers[job_type] = handler

    def has(self, job_type: str) -> bool:
        return job_type in self._handlers

    async def run(self, uow: UnitOfWork, job: Any, context: Any) -> None:
        handler = self._handlers.get(job.job_type)
        if handler is None:
            raise KeyError(f"no handler registered for job_type={job.job_type}")
        await handler(uow, job, context)


class SchedulerWorker:
    def __init__(
        self,
        session_factory: async_sessionmaker,
        runner: JobRunner,
        context: Any = None,
        *,
        poll_interval_seconds: float = 15.0,
        misfire_grace_seconds: int = 60,
        now_fn: Callable[[], dt.datetime] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._runner = runner
        self._context = context
        self._poll_interval = poll_interval_seconds
        self._grace = misfire_grace_seconds
        self._now = now_fn or (lambda: dt.datetime.now(UTC))
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="scheduler-worker")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        logger.info("scheduler_worker_started")
        while self._running:
            try:
                await self.sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - never die on one cycle
                logger.exception("scheduler_cycle_failed")
            await asyncio.sleep(self._poll_interval)

    async def sweep_once(self) -> int:
        """Runs every due job once. Returns the number of jobs dispatched."""
        now = self._now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        dispatched = 0
        uow = UnitOfWork(self._session_factory)
        async with uow:
            jobs = await uow.jobs.list_enabled()
            for job in jobs:
                next_run = job.next_run_at
                if next_run is None:
                    next_run = compute_next_run(job.cron_expr, after=now)
                elif next_run.tzinfo is None:
                    next_run = next_run.replace(tzinfo=UTC)
                if next_run is None:
                    continue
                if next_run > now:
                    continue
                if (now - next_run).total_seconds() > self._grace:
                    logger.warning(
                        "scheduler_job_misfired",
                        job_id=str(job.id),
                        job_type=job.job_type,
                        scheduled=next_run.isoformat(),
                    )
                    continue
                # Idempotency: only the first claim of this fire time runs it.
                run_key = f"run@{next_run:%Y%m%dT%H%M%S}"
                stored_at = next_run.astimezone(UTC).replace(tzinfo=None)
                claimed = await uow.jobs.record_run(
                    job.id, run_key=run_key, scheduled_at=stored_at
                )
                if claimed:
                    if self._runner.has(job.job_type):
                        try:
                            await self._runner.run(uow, job, self._context)
                        except Exception:  # noqa: BLE001 - one bad job must not block others
                            logger.exception(
                                "scheduler_job_failed",
                                job_id=str(job.id),
                                job_type=job.job_type,
                            )
                        dispatched += 1
                    else:
                        logger.warning(
                            "scheduler_no_handler",
                            job_id=str(job.id),
                            job_type=job.job_type,
                        )
                after = next_run + dt.timedelta(minutes=1)
                following = compute_next_run(job.cron_expr, after=after)
                if following is not None:
                    await uow.jobs.update_run_state(
                        job,
                        last_run_at=now.astimezone(UTC).replace(tzinfo=None),
                        next_run_at=following.replace(tzinfo=None),
                    )
            await uow.commit()
        return dispatched


__all__ = ["JobRunner", "JobHandler", "SchedulerWorker"]
