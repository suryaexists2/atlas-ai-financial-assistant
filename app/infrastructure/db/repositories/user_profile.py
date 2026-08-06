"""SQLAlchemy implementations of user/profile repositories."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import User, UserProfile
from app.domain.enums import OnboardingStatus
from app.domain.repositories import ProfileRepository, UserRepository


class SqlUserRepository(UserRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        return await self.session.get(User, user_id)

    async def get_by_telegram_id(self, telegram_id: int) -> User | None:
        result = await self.session.execute(select(User).where(User.telegram_id == telegram_id))
        return result.scalar_one_or_none()

    async def create(self, *, telegram_id: int, **fields: Any) -> User:
        user = User(telegram_id=telegram_id, **fields)
        self.session.add(user)
        await self.session.flush()
        return user

    async def update(self, user: User, **fields: Any) -> User:
        for key, value in fields.items():
            setattr(user, key, value)
        await self.session.flush()
        return user

    async def delete(self, user_id: uuid.UUID) -> None:
        user = await self.session.get(User, user_id)
        if user is not None:
            await self.session.delete(user)
            await self.session.flush()


class SqlProfileRepository(ProfileRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_user_id(self, user_id: uuid.UUID) -> UserProfile | None:
        result = await self.session.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def upsert(self, user_id: uuid.UUID, **fields: Any) -> UserProfile:
        profile = await self.get_by_user_id(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id, **fields)
            self.session.add(profile)
        else:
            for key, value in fields.items():
                setattr(profile, key, value)
        await self.session.flush()
        return profile

    async def set_onboarding(
        self,
        user_id: uuid.UUID,
        status: OnboardingStatus,
        context: dict[str, Any] | None = None,
    ) -> UserProfile:
        return await self.upsert(user_id, onboarding_status=status, onboarding_context=context)
