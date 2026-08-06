"""FastAPI dependency helpers."""

from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppError


def get_session_factory(request: Request):
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise AppError(
            "Application not ready: database session factory is unavailable.",
            code="service_unavailable",
            status_code=503,
        )
    return factory


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    session_factory = get_session_factory(request)
    async with session_factory() as session:
        yield session
