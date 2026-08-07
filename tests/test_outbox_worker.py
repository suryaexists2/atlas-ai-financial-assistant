"""OutboxWorker tests: drain, delivery, retry scheduling, exhaustion, status UX."""

import datetime as dt
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import update as sa_update

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


def make_status_sender(sent_ids: list) -> AsyncMock:
    sender = AsyncMock()

    def _send(**kwargs):
        if kwargs.get("capture_message_id"):
            return sent_ids.pop(0) if sent_ids else None
        return True

    sender.send.side_effect = _send
    sender.delete_message = AsyncMock(return_value=True)
    return sender


@pytest.mark.asyncio
async def test_status_sent_with_message_id_recorded(session_factory):
    status = await enqueue(
        session_factory,
        payload={"type": "status", "correlation_id": "c-s1", "text": "⏳ thinking..."},
    )
    sender = make_status_sender([77])
    worker = make_worker(session_factory, sender)

    await worker._drain_once()

    sender.send.assert_awaited_once()
    assert sender.send.await_args.kwargs["capture_message_id"] is True
    persisted = await get_message(session_factory, status.id)
    assert persisted.status == OutboundStatus.SENT
    assert persisted.payload["telegram_message_id"] == 77


@pytest.mark.asyncio
async def test_final_reply_deletes_status_before_send(session_factory):
    status = await enqueue(
        session_factory,
        payload={"type": "status", "correlation_id": "c-s2", "text": "⏳ thinking..."},
    )
    final = await enqueue(
        session_factory,
        payload={"type": "text", "text": "final answer", "correlation_id": "c-s2"},
    )
    deleted = []
    sender = make_status_sender([101])

    def _record_delete(**kwargs):
        deleted.append(kwargs["message_id"])
        return True

    sender.delete_message.side_effect = _record_delete
    worker = make_worker(session_factory, sender)

    await worker._drain_once()

    assert deleted == [101]  # status bubble removed before final reply sent
    assert sender.send.await_count == 2
    assert (await get_message(session_factory, final.id)).status == OutboundStatus.SENT
    persisted_status = await get_message(session_factory, status.id)
    assert persisted_status.status == OutboundStatus.FAILED
    assert persisted_status.last_error == "removed before final reply"


@pytest.mark.asyncio
async def test_final_failure_still_cleans_up_status(session_factory):
    status = await enqueue(
        session_factory,
        payload={"type": "status", "correlation_id": "c-s3", "text": "⏳ thinking..."},
    )
    final = await enqueue(
        session_factory,
        payload={"type": "text", "text": "answer", "correlation_id": "c-s3"},
    )
    deleted = []
    sender = AsyncMock()

    def _send(**kwargs):
        if kwargs.get("capture_message_id"):
            return 202
        return False  # final always fails -> max attempts

    sender.send.side_effect = _send
    sender.delete_message.side_effect = lambda **kw: deleted.append(kw["message_id"]) or True
    worker = make_worker(session_factory, sender)

    for _ in range(3):
        await worker._drain_once()

    assert deleted == [202]  # status bubble removed exactly once, final failed
    assert (await get_message(session_factory, final.id)).status == OutboundStatus.FAILED
    assert (await get_message(session_factory, status.id)).status == OutboundStatus.FAILED


@pytest.mark.asyncio
async def test_pending_status_superseded_by_final(session_factory):
    status = await enqueue(
        session_factory,
        payload={"type": "status", "correlation_id": "c-s4", "text": "⏳ thinking..."},
    )
    await enqueue(
        session_factory,
        payload={"type": "text", "text": "done", "correlation_id": "c-s4"},
    )
    sender = AsyncMock()
    # status send fails (capture -> False), final send succeeds
    sender.send.side_effect = lambda **kw: not kw.get("capture_message_id")
    worker = make_worker(session_factory, sender)

    await worker._drain_once()

    persisted = await get_message(session_factory, status.id)
    assert persisted.status == OutboundStatus.FAILED
    assert persisted.last_error == "superseded by final reply"
    sender.delete_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_statuses_are_per_correlation(session_factory):
    rows = []
    for corr in ("c-x", "c-y"):
        rows.append(
            await enqueue(
                session_factory,
                payload={"type": "status", "correlation_id": corr, "text": "⏳ thinking..."},
            )
        )
        rows.append(
            await enqueue(
                session_factory,
                payload={"type": "text", "text": f"answer-{corr}", "correlation_id": corr},
            )
        )
    deleted = []
    sender = make_status_sender([301, 302])
    sender.delete_message.side_effect = lambda **kw: deleted.append(kw["message_id"]) or True
    worker = make_worker(session_factory, sender)

    await worker._drain_once()

    assert deleted == [301, 302]  # each final deletes only its own bubble
    assert sender.send.await_count == 4
    for i, row in enumerate(rows):
        persisted = await get_message(session_factory, row.id)
        if i % 2 == 0:  # status bubble: closed out after its final reply
            assert persisted.status == OutboundStatus.FAILED
            assert persisted.last_error == "removed before final reply"
        else:  # final reply: delivered
            assert persisted.status == OutboundStatus.SENT


@pytest.mark.asyncio
async def test_expired_status_cleaned_up(session_factory):
    status = await enqueue(
        session_factory,
        payload={"type": "status", "correlation_id": "c-stale", "text": "⏳ thinking..."},
    )
    deleted = []
    sender = make_status_sender([777])
    sender.delete_message.side_effect = lambda **kw: deleted.append(kw["message_id"]) or True
    worker = make_worker(session_factory, sender, status_ttl_seconds=600.0)

    await worker._drain_once()  # delivered -> SENT with telegram id 777

    uow = UnitOfWork(session_factory)
    async with uow:
        await uow.session.execute(
            sa_update(OutboundMessage)
            .where(OutboundMessage.id == status.id)
            .values(created_at=dt.datetime.now(dt.UTC) - dt.timedelta(hours=2))
        )
        await uow.commit()

    await worker._drain_once()  # TTL sweep closes the orphaned bubble

    assert deleted == [777]  # stale bubble removed from chat
    persisted = await get_message(session_factory, status.id)
    assert persisted.status == OutboundStatus.FAILED
    assert persisted.last_error == "status expired"
