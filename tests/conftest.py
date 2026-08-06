"""Shared fixtures incl. Telegram-specific test helpers."""

import uuid

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.infrastructure.db.base import Base
from app.infrastructure.db.session import build_session_factory
from app.infrastructure.db.uow import UnitOfWork
from app.interfaces.api.deps import get_db
from app.main import create_app


@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(db_engine):
    return build_session_factory(db_engine)


@pytest_asyncio.fixture
async def uow(session_factory):
    return UnitOfWork(session_factory)


@pytest_asyncio.fixture
async def client(session_factory):
    settings = Settings(app_env="test", database_url="sqlite+aiosqlite://")
    app = create_app(settings)

    async def _get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def demo_user(uow, session_factory):
    telegram_id = 111111
    async with uow:
        user = await uow.users.create(
            telegram_id=telegram_id,
            username="tester",
            timezone="UTC",
        )
        await uow.profiles.upsert(user.id, role="Investor", briefing_time="08:00")
        await uow.commit()
    return {"user_id": user.id, "telegram_id": telegram_id}


# ---- Telegram test payload helpers ------------------------------------------


def tg_text_update(*, update_id: int, chat_id: int, message_id: int, text: str) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "date": 1780000000,
            "from": {"id": chat_id, "is_bot": False, "first_name": "Tester"},
            "chat": {"id": chat_id, "type": "private", "first_name": "Tester"},
            "text": text,
        },
    }


def tg_voice_update(*, chat_id: int, message_id: int, file_id: str = "f123") -> dict:
    return {
        "update_id": 999,
        "message": {
            "message_id": message_id,
            "date": 1780000000,
            "from": {"id": chat_id, "is_bot": False, "first_name": "Tester"},
            "chat": {"id": chat_id, "type": "private", "first_name": "Tester"},
            "voice": {
                "file_id": file_id,
                "file_unique_id": "fu-123",
                "duration": 5,
                "mime_type": "audio/ogg",
                "file_size": 2048,
            },
        },
    }


def tg_photo_update(*, chat_id: int, message_id: int, big_id: str = "ph_big") -> dict:
    return {
        "update_id": 998,
        "message": {
            "message_id": message_id,
            "date": 1780000000,
            "from": {"id": chat_id, "is_bot": False, "first_name": "Tester"},
            "chat": {"id": chat_id, "type": "private", "first_name": "Tester"},
            "photo": [
                {
                    "file_id": "ph_small",
                    "file_unique_id": "fu_small",
                    "file_size": 100,
                    "width": 10,
                    "height": 10,
                },
                {
                    "file_id": big_id,
                    "file_unique_id": "fu_big",
                    "file_size": 9000,
                    "width": 300,
                    "height": 300,
                },
            ],
            "caption": "check this chart",
        },
    }


def new_uid() -> uuid.UUID:
    return uuid.uuid4()
