"""Outbox worker: drains durable outbound queue and delivers via TelegramSender.

Runs as a background task (lifespan or the polling runner). Delivers in-band
so a restart never loses pending notifications: rows stay `pending` until the
API call succeeds, and failed rows are retried with exponential backoff until
`max_attempts` is reached.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt

from app.core.logging import get_logger
from app.infrastructure.db.session import async_sessionmaker
from app.infrastructure.db.uow import UnitOfWork
from app.infrastructure.telegram.sender import TelegramSender

logger = get_logger(__name__)


class OutboxWorker:
    def __init__(
        self,
        session_factory: async_sessionmaker,
        sender: TelegramSender,
        *,
        poll_interval_seconds: float = 0.5,
        max_attempts: int = 5,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 300.0,
    ) -> None:
        self._session_factory = session_factory
        self._sender = sender
        self._poll_interval = poll_interval_seconds
        self._max_attempts = max_attempts
        self._retry_base = retry_base_seconds
        self._retry_max = retry_max_seconds
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="outbox-worker")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        logger.info("outbox_worker_started")
        while self._running:
            try:
                await self._drain_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - keep the worker alive
                logger.exception("outbox_worker_cycle_failed")
            await asyncio.sleep(self._poll_interval)

    async def _drain_once(self) -> None:
        uow = UnitOfWork(self._session_factory)
        async with uow:
            messages = await uow.outbox.claim_due(limit=20)
            if not messages:
                return
            for message in messages:
                chat_id = message.chat_id
                payload = message.payload
                if chat_id is None:
                    await uow.outbox.mark_failed(
                        message, error="missing chat_id", next_retry_at=None
                    )
                    continue

                delivered = await self._sender.send(chat_id=chat_id, payload=payload)
                if delivered:
                    await uow.outbox.mark_sent(message)
                    continue

                # mark_failed increments `attempt`; plan the retry from the
                # value this row will hold after the update.
                next_attempt = message.attempt + 1
                if next_attempt >= self._max_attempts:
                    await uow.outbox.mark_failed(
                        message, error="max attempts reached", next_retry_at=None
                    )
                    logger.warning(
                        "outbox_delivery_exhausted",
                        outbound_id=str(message.id),
                        chat_id=chat_id,
                        correlation_id=payload.get("correlation_id"),
                    )
                else:
                    delay = min(self._retry_max, self._retry_base * (2 ** (next_attempt - 1)))
                    next_retry = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=delay)
                    await uow.outbox.mark_failed(
                        message,
                        error="transient failure",
                        next_retry_at=next_retry,
                    )
                    logger.info(
                        "outbox_retry_scheduled",
                        outbound_id=str(message.id),
                        attempt=next_attempt,
                        next_retry_at=next_retry.isoformat(),
                        correlation_id=payload.get("correlation_id"),
                    )
