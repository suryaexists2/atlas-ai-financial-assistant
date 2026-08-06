"""OutboxWorker tests: drain, delivery, retry scheduling, exhaustion."""

from unittest.mock import AsyncMock

import pytest

from app.domain.entities import OutboundMessage, OutboundStatus
from app.infrastructure.db.uow import UnitOfWork
from app.infrastructure.telegram.outbox_worker import OutboxWorker


async def enqueue(
    session_factory, *, chat_id: int = 555, payload: dict | None = None
) -> OutboundMessage:
    uow = UnitOfWork(session_factory)
    async with uow:
        message = await uow.outbox.enqueue(
            chat_id=chat_id,
            payload=payload or {"type": "text", "text": "hi", "correlation_id": "c-1"},
        )
        await uow.commit()
        return message


def make_worker(session_factory, sender, **kwargs) -> OutboxWorker:
    return OutboxWorker(
        session_factory,
        sender,
        poll_interval_seconds=999,  # never loop during tests; we call _drain_once directly
        retry_base_seconds=0.0,  # retries become due immediately, so drains can be looped
        retry_max_seconds=0.0,
        max_attempts=3,
        **kwargs,
    )


async def get_message(session_factory, message_id) -> OutboundMessage:
    uow = UnitOfWork(session_factory)
    async with uow:
        return await uow.session.get(OutboundMessage, message_id)


@pytest.mark.asyncio
async def test_drain_sends_and_marks_sent(session_factory):
    message = await enqueue(session_factory)
    sender = AsyncMock()
    sender.send.return_value = True
    worker = make_worker(session_factory, sender)

    await worker._drain_once()

    sender.send.assert_awaited_once()
    kwargs = sender.send.await_args.kwargs
    assert kwargs["chat_id"] == 555
    assert kwargs["payload"]["text"] == "hi"

    persisted = await get_message(session_factory, message.id)
    assert persisted.status == OutboundStatus.SENT
    assert persisted.sent_at is not None


@pytest.mark.asyncio
async def test_transient_failure_schedules_retry(session_factory):
    message = await enqueue(session_factory)
    sender = AsyncMock()
    sender.send.return_value = False
    worker = make_worker(session_factory, sender)

    await worker._drain_once()

    persisted = await get_message(session_factory, message.id)
    assert persisted.status == OutboundStatus.PENDING
    assert persisted.attempt == 1
    assert persisted.next_retry_at is not None  # retry scheduled
    assert persisted.last_error == "transient failure"


@pytest.mark.asyncio
async def test_attempts_exhausted_marks_failed(session_factory):
    message = await enqueue(session_factory)
    sender = AsyncMock()
    sender.send.return_value = False
    worker = make_worker(session_factory, sender)

    for _ in range(3):  # attempt 1,2 -> retry; attempt 3 -> failed
        await worker._drain_once()

    persisted = await get_message(session_factory, message.id)
    assert persisted.status == OutboundStatus.FAILED
    assert persisted.attempt == 3
    assert persisted.last_error == "max attempts reached"


@pytest.mark.asyncio
async def test_missing_chat_id_fails_immediately(session_factory):
    message = await enqueue(session_factory, chat_id=None)
    sender = AsyncMock()
    sender.send.return_value = True
    worker = make_worker(session_factory, sender)

    await worker._drain_once()

    sender.send.assert_not_awaited()
    persisted = await get_message(session_factory, message.id)
    assert persisted.status == OutboundStatus.FAILED
    assert persisted.last_error == "missing chat_id"


@pytest.mark.asyncio
async def test_retry_only_due_messages(session_factory):
    await enqueue(session_factory)
    sender = AsyncMock()
    sender.send.return_value = True
    worker = make_worker(session_factory, sender)

    await worker._drain_once()  # mark sent
    await worker._drain_once()  # queue empty now

    assert sender.send.await_count == 1  # not re-delivered


@pytest.mark.asyncio
async def test_start_stop_lifecycle(session_factory):
    sender = AsyncMock()
    sender.send.return_value = True
    worker = make_worker(session_factory, sender)
    worker.start()
    assert worker._task is not None
    await worker.stop()
    assert worker._task is None
