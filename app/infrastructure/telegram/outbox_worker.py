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
from typing import Any

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
        status_ttl_seconds: float = 600.0,
    ) -> None:
        self._session_factory = session_factory
        self._sender = sender
        self._poll_interval = poll_interval_seconds
        self._max_attempts = max_attempts
        self._retry_base = retry_base_seconds
        self._retry_max = retry_max_seconds
        self._status_ttl = status_ttl_seconds
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
            await self._expire_stale_statuses(uow)
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

                if payload.get("type") == "status":
                    await self._deliver_status(uow, message, chat_id, payload)
                    continue

                delivered = await self._sender.send(chat_id=chat_id, payload=payload)
                correlation_id = payload.get("correlation_id")
                if delivered:
                    await self._cleanup_status(uow, chat_id, correlation_id)
                    await uow.outbox.mark_sent(message)
                    continue

                # mark_failed increments `attempt`; plan the retry from the
                # value this row will hold after the update.
                next_attempt = message.attempt + 1
                if next_attempt >= self._max_attempts:
                    await self._cleanup_status(uow, chat_id, correlation_id)
                    await uow.outbox.mark_failed(
                        message, error="max attempts reached", next_retry_at=None
                    )
                    logger.warning(
                        "outbox_delivery_exhausted",
                        outbound_id=str(message.id),
                        chat_id=chat_id,
                        correlation_id=correlation_id,
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
                        correlation_id=correlation_id,
                    )

    async def _deliver_status(
        self, uow: UnitOfWork, message: Any, chat_id: int, payload: dict[str, Any]
    ) -> None:
        """Sends a temporary status message and remembers its Telegram id."""
        sent_id = await self._sender.send(
            chat_id=chat_id, payload=payload, capture_message_id=True
        )
        if isinstance(sent_id, int) and not isinstance(sent_id, bool):
            # bool is an int subclass: a False (failure) must not read as a message id
            message.payload = {**payload, "telegram_message_id": sent_id}
            await uow.outbox.mark_sent(message)
            return
        next_attempt = message.attempt + 1
        if next_attempt >= self._max_attempts:
            await uow.outbox.mark_failed(
                message, error="status delivery exhausted", next_retry_at=None
            )
            return
        delay = min(self._retry_max, self._retry_base * (2 ** (next_attempt - 1)))
        await uow.outbox.mark_failed(
            message,
            error="transient failure",
            next_retry_at=dt.datetime.now(dt.UTC) + dt.timedelta(seconds=delay),
        )

    async def _cleanup_status(
        self, uow: UnitOfWork, chat_id: int, correlation_id: str | None
    ) -> None:
        """Deletes the delivered status message (if any) and cancels any pending
        status rows so a stale "thinking" bubble can never outlive the reply."""
        if not correlation_id:
            return
        status = await uow.outbox.get_sent_status(correlation_id)
        if status is not None:
            telegram_id = status.payload.get("telegram_message_id")
            if isinstance(telegram_id, int):
                await self._sender.delete_message(chat_id=chat_id, message_id=telegram_id)
            await uow.outbox.mark_failed(
                status, error="removed before final reply", next_retry_at=None
            )
        await uow.outbox.supersede_statuses(correlation_id)

    async def _expire_stale_statuses(self, uow: UnitOfWork) -> None:
        """TTL sweep: status messages that never got a final reply (e.g. the
        webhook died mid-turn) are deleted from Telegram and closed out."""
        older_than = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=self._status_ttl)
        stale = await uow.outbox.expire_statuses(older_than)
        for status in stale:
            telegram_id = status.payload.get("telegram_message_id")
            if isinstance(telegram_id, int):
                await self._sender.delete_message(
                    chat_id=status.chat_id, message_id=telegram_id
                )
            await uow.outbox.mark_failed(
                status, error="status expired", next_retry_at=None
            )
            logger.info(
                "outbox_status_expired",
                outbound_id=str(status.id),
                chat_id=status.chat_id,
            )
