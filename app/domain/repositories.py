"""Repository interfaces (ports).

Services depend on these abstractions, not on SQLAlchemy, so persistence can
be swapped or mocked freely. Each abstraction is small and focused.
"""

from __future__ import annotations

import abc
import datetime as dt
import uuid
from typing import Any

from app.domain.entities import (
    Alert,
    Conversation,
    IntegrationLink,
    Memory,
    Message,
    OutboundMessage,
    ScheduledJob,
    User,
    UserProfile,
    WatchlistItem,
)
from app.domain.enums import (
    DocumentStatus,
    IntegrationProvider,
    MessageRole,
    OnboardingStatus,
)


class UserRepository(abc.ABC):
    @abc.abstractmethod
    async def get_by_id(self, user_id: uuid.UUID) -> User | None: ...

    @abc.abstractmethod
    async def get_by_telegram_id(self, telegram_id: int) -> User | None: ...

    @abc.abstractmethod
    async def create(self, *, telegram_id: int, **fields: Any) -> User: ...

    @abc.abstractmethod
    async def update(self, user: User, **fields: Any) -> User: ...

    @abc.abstractmethod
    async def delete(self, user_id: uuid.UUID) -> None: ...


class ProfileRepository(abc.ABC):
    @abc.abstractmethod
    async def get_by_user_id(self, user_id: uuid.UUID) -> UserProfile | None: ...

    @abc.abstractmethod
    async def upsert(self, user_id: uuid.UUID, **fields: Any) -> UserProfile: ...

    @abc.abstractmethod
    async def set_onboarding(
        self,
        user_id: uuid.UUID,
        status: OnboardingStatus,
        context: dict[str, Any] | None = None,
    ) -> UserProfile: ...


class ConversationRepository(abc.ABC):
    @abc.abstractmethod
    async def create(self, user_id: uuid.UUID, *, title: str | None = None) -> Conversation: ...

    @abc.abstractmethod
    async def get_by_id(self, conversation_id: uuid.UUID) -> Conversation | None: ...

    @abc.abstractmethod
    async def list_for_user(self, user_id: uuid.UUID, limit: int = 20) -> list[Conversation]: ...

    @abc.abstractmethod
    async def add_message(
        self,
        conversation_id: uuid.UUID,
        *,
        role: MessageRole,
        content: str | None,
        content_type: Any = None,
        media_meta: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> Message: ...

    @abc.abstractmethod
    async def update_message(
        self,
        message_id: uuid.UUID,
        *,
        content: str | None = None,
        media_meta: dict[str, Any] | None = None,
    ) -> Message | None: ...

    @abc.abstractmethod
    async def list_messages(self, conversation_id: uuid.UUID, limit: int = 50) -> list[Message]: ...


class WatchlistRepository(abc.ABC):
    @abc.abstractmethod
    async def add(
        self, user_id: uuid.UUID, *, symbol: str, name: str | None, sector: str | None
    ) -> WatchlistItem: ...

    @abc.abstractmethod
    async def get_by_symbol(self, user_id: uuid.UUID, symbol: str) -> WatchlistItem | None: ...

    @abc.abstractmethod
    async def list_active(self, user_id: uuid.UUID) -> list[WatchlistItem]: ...

    @abc.abstractmethod
    async def deactivate(self, item: WatchlistItem) -> WatchlistItem: ...


class AlertRepository(abc.ABC):
    @abc.abstractmethod
    async def create(self, user_id: uuid.UUID, **fields: Any) -> Alert: ...

    @abc.abstractmethod
    async def get_by_id(self, alert_id: uuid.UUID) -> Alert | None: ...

    @abc.abstractmethod
    async def list_enabled(self, user_id: uuid.UUID | None = None) -> list[Alert]: ...

    @abc.abstractmethod
    async def update(self, alert: Alert, **fields: Any) -> Alert: ...

    @abc.abstractmethod
    async def delete(self, alert: Alert) -> None: ...


class DocumentRepository(abc.ABC):
    @abc.abstractmethod
    async def create(self, user_id: uuid.UUID, **fields: Any) -> Any: ...

    @abc.abstractmethod
    async def get_by_id(self, document_id: uuid.UUID) -> Any | None: ...

    @abc.abstractmethod
    async def list_for_user(self, user_id: uuid.UUID, limit: int = 50) -> list[Any]: ...

    @abc.abstractmethod
    async def update_status(self, document: Any, status: DocumentStatus, **meta: Any) -> Any: ...


class MemoryRepository(abc.ABC):
    @abc.abstractmethod
    async def get_active(self, user_id: uuid.UUID, memory_key: str) -> Memory | None: ...

    @abc.abstractmethod
    async def list_active(self, user_id: uuid.UUID, limit: int = 200) -> list[Memory]: ...

    @abc.abstractmethod
    async def upsert_observation(
        self,
        user_id: uuid.UUID,
        *,
        memory_key: str,
        value: dict[str, Any] | None,
        summary: str | None,
        confidence: float,
        source_turn_id: uuid.UUID | None = None,
    ) -> Memory: ...

    @abc.abstractmethod
    async def supersede(
        self, user_id: uuid.UUID, memory_key: str, archived: bool = False
    ) -> int: ...


class JobRepository(abc.ABC):
    @abc.abstractmethod
    async def create(
        self,
        *,
        job_type: str,
        cron_expr: str,
        user_id: uuid.UUID | None = None,
        params: dict[str, Any] | None = None,
        timezone: str = "UTC",
    ) -> ScheduledJob: ...

    @abc.abstractmethod
    async def get_by_id(self, job_id: uuid.UUID) -> ScheduledJob | None: ...

    @abc.abstractmethod
    async def list_enabled(self) -> list[ScheduledJob]: ...

    @abc.abstractmethod
    async def delete(self, job: ScheduledJob) -> None: ...

    @abc.abstractmethod
    async def lock_for_run(self, job_id: uuid.UUID) -> ScheduledJob | None: ...

    @abc.abstractmethod
    async def update_run_state(
        self, job: ScheduledJob, *, last_run_at: dt.datetime, next_run_at: dt.datetime
    ) -> ScheduledJob: ...

    @abc.abstractmethod
    async def record_run(
        self,
        job_id: uuid.UUID,
        *,
        run_key: str,
        scheduled_at: dt.datetime,
    ) -> bool: ...


class OutboxRepository(abc.ABC):
    @abc.abstractmethod
    async def enqueue(
        self, *, chat_id: int | None = None, payload: dict[str, Any], priority: int = 0
    ) -> OutboundMessage: ...

    @abc.abstractmethod
    async def claim_due(self, limit: int = 20) -> list[OutboundMessage]: ...

    @abc.abstractmethod
    async def mark_sent(self, message: OutboundMessage) -> OutboundMessage: ...

    @abc.abstractmethod
    async def mark_failed(
        self, message: OutboundMessage, *, error: str, next_retry_at: dt.datetime | None
    ) -> OutboundMessage: ...


class IntegrationRepository(abc.ABC):
    @abc.abstractmethod
    async def upsert(
        self,
        user_id: uuid.UUID,
        *,
        provider: IntegrationProvider,
        access_token: str,
        refresh_token: str | None = None,
        scopes: list[str] | None = None,
        expires_at: dt.datetime | None = None,
    ) -> IntegrationLink: ...

    @abc.abstractmethod
    async def get_by_provider(
        self, user_id: uuid.UUID, provider: IntegrationProvider
    ) -> IntegrationLink | None: ...

    @abc.abstractmethod
    async def delete(self, link: IntegrationLink) -> None: ...


class IngestLedgerRepository(abc.ABC):
    """Dedup ledger for raw provider updates (update_id + message_id)."""

    @abc.abstractmethod
    async def record(
        self,
        *,
        update_id: int,
        chat_id: int,
        message_id: int | None,
        source: str,
        correlation_id: str,
        user_id: uuid.UUID | None = None,
    ) -> bool:
        """Persists the update; returns False if update_id or (chat, message) is a duplicate."""


__all__ = [
    "UserRepository",
    "ProfileRepository",
    "ConversationRepository",
    "WatchlistRepository",
    "AlertRepository",
    "DocumentRepository",
    "MemoryRepository",
    "JobRepository",
    "OutboxRepository",
    "IntegrationRepository",
    "IngestLedgerRepository",
]
