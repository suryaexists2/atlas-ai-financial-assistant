"""Memory repository tests: update-not-duplicate, confidence, supersede (M1)."""

import pytest


@pytest.mark.asyncio
async def test_upsert_creates_then_updates_without_duplicates(uow, demo_user):
    user_id = demo_user["user_id"]

    async with uow:
        first = await uow.memories.upsert_observation(
            user_id,
            memory_key="interest:ai",
            value={"label": "AI"},
            summary="Interested in AI",
            confidence=0.6,
        )
        await uow.commit()
        assert first.version == 1

    async with uow:
        second = await uow.memories.upsert_observation(
            user_id,
            memory_key="interest:ai",
            value={"label": "AI / Semis"},
            summary="Interested in AI and semiconductors",
            confidence=0.9,
        )
        await uow.commit()
        assert second.version == 2
        assert second.value == {"label": "AI / Semis"}

    async with uow:
        active = await uow.memories.list_active(user_id)
        assert len(active) == 1  # no duplicates
        assert active[0].confidence > 0.6  # reinforced by recency weighting
        assert active[0].last_seen_at is not None


@pytest.mark.asyncio
async def test_confidence_reinforcement_weights_toward_new_value(uow, demo_user):
    user_id = demo_user["user_id"]

    async with uow:
        await uow.memories.upsert_observation(
            user_id, memory_key="watch:aapl", value={"symbol": "AAPL"}, summary="", confidence=1.0
        )
        await uow.commit()

    async with uow:
        reinforced = await uow.memories.upsert_observation(
            user_id, memory_key="watch:aapl", value={"symbol": "AAPL"}, summary="", confidence=0.5
        )
        await uow.commit()
        # 0.6*1.0 + 0.4*0.5 = 0.8 — stays anchored to the stronger evidence
        assert reinforced.confidence == pytest.approx(0.8, abs=1e-4)


@pytest.mark.asyncio
async def test_supersede_marks_old_version_inactive(uow, demo_user):
    user_id = demo_user["user_id"]

    async with uow:
        await uow.memories.upsert_observation(
            user_id, memory_key="role", value={"role": "Investor"}, summary="", confidence=0.8
        )
        await uow.commit()

    async with uow:
        changed = await uow.memories.supersede(user_id, "role")
        await uow.commit()
        assert changed == 1

    async with uow:
        assert await uow.memories.get_active(user_id, "role") is None
        assert len(await uow.memories.list_active(user_id)) == 0

    # A new observation after supersede creates a fresh active version
    async with uow:
        fresh = await uow.memories.upsert_observation(
            user_id, memory_key="role", value={"role": "Founder"}, summary="", confidence=0.7
        )
        await uow.commit()
        assert fresh.version == 1
        assert fresh.status == "active"


@pytest.mark.asyncio
async def test_supersede_archived_mode(uow, demo_user):
    user_id = demo_user["user_id"]
    from app.domain.enums import MemoryStatus

    async with uow:
        await uow.memories.upsert_observation(
            user_id, memory_key="topic:macro", value={}, summary="", confidence=0.5
        )
        await uow.commit()

    async with uow:
        await uow.memories.supersede(user_id, "topic:macro", archived=True)
        await uow.commit()

    async with uow:
        from sqlalchemy import select

        from app.domain.entities import Memory

        result = await uow.session.execute(
            select(Memory).where(Memory.user_id == user_id, Memory.memory_key == "topic:macro")
        )
        memory = result.scalar_one()
        assert memory.status == MemoryStatus.ARCHIVED


@pytest.mark.asyncio
async def test_memory_isolation_between_users(uow, demo_user, session_factory):
    async with uow:
        other = await uow.users.create(telegram_id=888, username="other2")
        await uow.commit()
        other_id = other.id

    async with uow:
        await uow.memories.upsert_observation(
            other_id, memory_key="interest:ai", value={}, summary="", confidence=0.5
        )
        await uow.commit()

    async with uow:
        assert await uow.memories.get_active(demo_user["user_id"], "interest:ai") is None
        assert len(await uow.memories.list_active(demo_user["user_id"])) == 0
        assert await uow.memories.get_active(other_id, "interest:ai") is not None
