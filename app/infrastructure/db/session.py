"""Async engine / session management.

Engine is built from `Settings` and owned by the FastAPI lifespan; the
`get_db` FastAPI dependency yields a session per request/background task.
Tests create their own engine and override `get_db`.
"""

from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings
from app.infrastructure.db.base import Base


def asyncpg_connect_args(database_url: str) -> dict[str, Any]:
    """Driver connect args for a given URL.

    Supabase (and similar managed Postgres) routes through pgbouncer in
    transaction mode, which does not support asyncpg's prepared statement
    cache; disabling it prevents DuplicatePreparedStatementError on pooled
    transactions. Must be applied to every engine built from the URL —
    including Alembic's — or migrations crash the same way the app did.
    """
    if database_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {"statement_cache_size": 0}


def build_engine(settings: Settings) -> AsyncEngine:
    kwargs: dict[str, Any] = {"echo": settings.debug}
    kwargs["connect_args"] = asyncpg_connect_args(settings.database_url)
    if not settings.database_url.startswith("sqlite"):
        kwargs.update(
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=settings.db_pool_pre_ping,
        )
    return create_async_engine(settings.database_url, **kwargs)


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    """Create all tables. Used for tests and fresh local dev; production uses Alembic."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine(engine: AsyncEngine) -> None:
    await engine.dispose()
