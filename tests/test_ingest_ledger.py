"""Ingest ledger tests: update_id + (chat, message) dedup guarantees."""

import pytest
from sqlalchemy import func, select

from app.domain.entities import IngestedUpdate
from app.infrastructure.db.uow import UnitOfWork


@pytest.mark.asyncio
async def test_same_update_id_rejected(session_factory):
    uow = UnitOfWork(session_factory)
    async with uow:
        first = await uow.ingest.record(
            update_id=1, chat_id=100, message_id=1, source="webhook", correlation_id="c-1"
        )
        second = await uow.ingest.record(
            update_id=1, chat_id=200, message_id=2, source="webhook", correlation_id="c-2"
        )
        assert first is True
        assert second is False

        result = await uow.session.execute(select(func.count()).select_from(IngestedUpdate))
        assert result.scalar_one() == 1


@pytest.mark.asyncio
async def test_same_chat_and_message_id_rejected(session_factory):
    uow = UnitOfWork(session_factory)
    async with uow:
        first = await uow.ingest.record(
            update_id=10, chat_id=300, message_id=7, source="webhook", correlation_id="c-3"
        )
        replay = await uow.ingest.record(
            update_id=11, chat_id=300, message_id=7, source="polling", correlation_id="c-4"
        )
        assert first is True
        assert replay is False


@pytest.mark.asyncio
async def test_distinct_messages_accepted(session_factory):
    uow = UnitOfWork(session_factory)
    async with uow:
        for i in range(5):
            assert (
                await uow.ingest.record(
                    update_id=100 + i,
                    chat_id=400,
                    message_id=i,
                    source="polling",
                    correlation_id=f"c-{i}",
                )
                is True
            )

        result = await uow.session.execute(select(func.count()).select_from(IngestedUpdate))
        assert result.scalar_one() == 5
