"""Minimal Unit-of-Work for transaction management.

A UoW owns one session and a set of repositories, providing atomic
commit/rollback semantics to application services:

    async with unit_of_work() as uow:
        user = await uow.users.create(telegram_id=...)
        await uow.commit()
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.infrastructure.db.repositories.conversation_market import (
    SqlAlertRepository,
    SqlConversationRepository,
    SqlWatchlistRepository,
)
from app.infrastructure.db.repositories.memory_jobs_outbox import (
    SqlDocumentRepository,
    SqlIngestLedgerRepository,
    SqlIntegrationRepository,
    SqlJobRepository,
    SqlMemoryRepository,
    SqlOAuthFlowRepository,
    SqlOutboxRepository,
)
from app.infrastructure.db.repositories.user_profile import (
    SqlProfileRepository,
    SqlUserRepository,
)


class UnitOfWork:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self.session: AsyncSession
        self.users: SqlUserRepository
        self.profiles: SqlProfileRepository
        self.conversations: SqlConversationRepository
        self.watchlist: SqlWatchlistRepository
        self.alerts: SqlAlertRepository
        self.documents: SqlDocumentRepository
        self.memories: SqlMemoryRepository
        self.jobs: SqlJobRepository
        self.outbox: SqlOutboxRepository
        self.integrations: SqlIntegrationRepository
        self.oauth_flows: SqlOAuthFlowRepository
        self.ingest: SqlIngestLedgerRepository

    def _bind_repositories(self) -> None:
        self.users = SqlUserRepository(self.session)
        self.profiles = SqlProfileRepository(self.session)
        self.conversations = SqlConversationRepository(self.session)
        self.watchlist = SqlWatchlistRepository(self.session)
        self.alerts = SqlAlertRepository(self.session)
        self.documents = SqlDocumentRepository(self.session)
        self.memories = SqlMemoryRepository(self.session)
        self.jobs = SqlJobRepository(self.session)
        self.outbox = SqlOutboxRepository(self.session)
        self.integrations = SqlIntegrationRepository(self.session)
        self.oauth_flows = SqlOAuthFlowRepository(self.session)
        self.ingest = SqlIngestLedgerRepository(self.session)

    async def __aenter__(self) -> UnitOfWork:
        self.session = self._session_factory()
        self._bind_repositories()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                await self.session.commit()
            else:
                await self.session.rollback()
        finally:
            await self.session.close()

    async def commit(self) -> None:
        await self.session.commit()

    async def rollback(self) -> None:
        await self.session.rollback()
