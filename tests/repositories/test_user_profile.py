"""User and profile repository CRUD tests (M1)."""

import pytest
from sqlalchemy.exc import IntegrityError


@pytest.mark.asyncio
async def test_create_and_get_by_telegram_id(uow):
    async with uow:
        user = await uow.users.create(telegram_id=999, username="alice", timezone="UTC")
        await uow.commit()
        assert user.id is not None
        assert user.telegram_id == 999

        fetched = await uow.users.get_by_telegram_id(999)
        assert fetched is not None
        assert fetched.username == "alice"

        missing = await uow.users.get_by_telegram_id(12345)
        assert missing is None


@pytest.mark.asyncio
async def test_update_and_delete_user(uow):
    async with uow:
        user = await uow.users.create(telegram_id=222, username="bob")
        await uow.commit()

    async with uow:
        user = await uow.users.get_by_telegram_id(222)
        updated = await uow.users.update(user, timezone="America/New_York")
        await uow.commit()
        assert updated.timezone == "America/New_York"

    async with uow:
        user = await uow.users.get_by_telegram_id(222)
        await uow.users.delete(user.id)
        await uow.commit()
        assert await uow.users.get_by_telegram_id(222) is None


@pytest.mark.asyncio
async def test_unique_telegram_id_enforced(uow):
    async with uow:
        await uow.users.create(telegram_id=333)
        await uow.commit()
    with pytest.raises(IntegrityError):
        async with uow:
            await uow.users.create(telegram_id=333)
            await uow.commit()


@pytest.mark.asyncio
async def test_profile_upsert_idempotent(uow, demo_user):
    user_id = demo_user["user_id"]

    async with uow:
        profile = await uow.profiles.upsert(user_id, role="Analyst", interests=["AI"])
        await uow.commit()
        assert profile.user_id == user_id
        first_id = profile.id

    async with uow:
        profile = await uow.profiles.upsert(user_id, role="Analyst", interests=["AI", "Chips"])
        await uow.commit()
        assert profile.id == first_id
        assert profile.interests == ["AI", "Chips"]


@pytest.mark.asyncio
async def test_onboarding_state_transition(uow, demo_user):
    user_id = demo_user["user_id"]
    from app.domain.enums import OnboardingStatus

    async with uow:
        await uow.profiles.set_onboarding(user_id, OnboardingStatus.IN_PROGRESS, {"step": 1})
        await uow.commit()

    async with uow:
        profile = await uow.profiles.get_by_user_id(user_id)
        assert profile.onboarding_status == OnboardingStatus.IN_PROGRESS
        assert profile.onboarding_context == {"step": 1}
