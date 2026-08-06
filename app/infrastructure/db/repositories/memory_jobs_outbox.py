"""SQLAlchemy implementations of memory, job, outbox, document and integration repos."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import (
    Document,
    IngestedUpdate,
    IntegrationLink,
    JobEvent,
    Memory,
    OutboundMessage,
    ScheduledJob,
)
from app.domain.enums import (
    DocumentStatus,
    IntegrationProvider,
    MemoryStatus,
    OutboundStatus,
)
from app.domain.repositories import (
    DocumentRepository,
    IngestLedgerRepository,
    IntegrationRepository,
    JobRepository,
    MemoryRepository,
    OutboxRepository,
)


class SqlMemoryRepository(MemoryRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_active(self, user_id: uuid.UUID, memory_key: str) -> Memory | None:
        result = await self.session.execute(
            select(Memory).where(
                Memory.user_id == user_id,
                Memory.memory_key == memory_key,
                Memory.status == MemoryStatus.ACTIVE,
            )
        )
        return result.scalar_one_or_none()

    async def list_active(self, user_id: uuid.UUID, limit: int = 200) -> list[Memory]:
        result = await self.session.execute(
            select(Memory)
            .where(Memory.user_id == user_id, Memory.status == MemoryStatus.ACTIVE)
            .order_by(Memory.updated_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def upsert_observation(
        self,
        user_id: uuid.UUID,
        *,
        memory_key: str,
        value: dict[str, Any] | None,
        summary: str | None,
        confidence: float,
        source_turn_id: uuid.UUID | None = None,
    ) -> Memory:
        """Update-not-duplicate semantics with recency-weighted confidence.

        A repeated observation reinforces the memory (confidence rises toward
        the new value); the version counter and last_seen_at timestamp make
        the update auditable.
        """
        existing = await self.get_active(user_id, memory_key)
        if existing is not None:
            merged = min(1.0, existing.confidence * 0.6 + confidence * 0.4)
            existing.value = value
            existing.summary = summary
            existing.confidence = round(merged, 4)
            existing.version += 1
            existing.last_seen_at = dt.datetime.now(dt.UTC)
            if source_turn_id is not None:
                existing.source_turn_id = source_turn_id
            memory = existing
        else:
            memory = Memory(
                user_id=user_id,
                memory_key=memory_key,
                value=value,
                summary=summary,
                confidence=min(1.0, confidence),
                source_turn_id=source_turn_id,
                last_seen_at=dt.datetime.now(dt.UTC),
                status=MemoryStatus.ACTIVE,
            )
            self.session.add(memory)
        await self.session.flush()
        return memory

    async def supersede(self, user_id: uuid.UUID, memory_key: str, archived: bool = False) -> int:
        new_status = MemoryStatus.ARCHIVED if archived else MemoryStatus.SUPERSEDED
        result = await self.session.execute(
            update(Memory)
            .where(
                Memory.user_id == user_id,
                Memory.memory_key == memory_key,
                Memory.status == MemoryStatus.ACTIVE,
            )
            .values(status=new_status)
        )
        await self.session.flush()
        return result.rowcount or 0


class SqlJobRepository(JobRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        job_type: str,
        cron_expr: str,
        user_id: uuid.UUID | None = None,
        params: dict[str, Any] | None = None,
        timezone: str = "UTC",
    ) -> ScheduledJob:
        job = ScheduledJob(
            job_type=job_type,
            cron_expr=cron_expr,
            user_id=user_id,
            params=params,
            timezone=timezone,
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def get_by_id(self, job_id: uuid.UUID) -> ScheduledJob | None:
        return await self.session.get(ScheduledJob, job_id)

    async def list_enabled(self) -> list[ScheduledJob]:
        result = await self.session.execute(
            select(ScheduledJob).where(ScheduledJob.enabled.is_(True))
        )
        return list(result.scalars().all())

    async def delete(self, job: ScheduledJob) -> None:
        await self.session.delete(job)
        await self.session.flush()

    async def lock_for_run(self, job_id: uuid.UUID) -> ScheduledJob | None:
        result = await self.session.execute(
            select(ScheduledJob)
            .where(ScheduledJob.id == job_id, ScheduledJob.enabled.is_(True))
            .with_for_update()
        )
        return result.scalar_one_or_none()

    async def update_run_state(
        self, job: ScheduledJob, *, last_run_at: dt.datetime, next_run_at: dt.datetime
    ) -> ScheduledJob:
        job.last_run_at = last_run_at
        job.next_run_at = next_run_at
        await self.session.flush()
        return job

    async def record_run(
        self,
        job_id: uuid.UUID,
        *,
        run_key: str,
        scheduled_at: dt.datetime,
    ) -> bool:
        """Idempotent ledger insert. Returns True only on first delivery of this run."""
        values = {
            "job_id": job_id,
            "run_key": run_key,
            "scheduled_at": scheduled_at,
            "executed_at": dt.datetime.now(dt.UTC),
        }
        dialect = self.session.sync_session.bind.dialect.name
        if dialect == "postgresql":
            stmt = (
                pg_insert(JobEvent)
                .values(**values)
                .on_conflict_do_nothing(index_elements=["job_id", "run_key"])
            )
        else:
            stmt = (
                sqlite_insert(JobEvent)
                .values(**values)
                .on_conflict_do_nothing(index_elements=["job_id", "run_key"])
            )
        result = await self.session.execute(stmt)
        await self.session.flush()
        return result.rowcount == 1


class SqlOutboxRepository(OutboxRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def enqueue(
        self, *, chat_id: int | None = None, payload: dict[str, Any], priority: int = 0
    ) -> OutboundMessage:
        message = OutboundMessage(
            chat_id=chat_id, payload=payload, priority=priority, status=OutboundStatus.PENDING
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def claim_due(self, limit: int = 20) -> list[OutboundMessage]:
        now = dt.datetime.now(dt.UTC)
        result = await self.session.execute(
            select(OutboundMessage)
            .where(
                OutboundMessage.status == OutboundStatus.PENDING,
                or_(OutboundMessage.next_retry_at.is_(None), OutboundMessage.next_retry_at <= now),
            )
            .order_by(OutboundMessage.priority.desc(), OutboundMessage.created_at.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return list(result.scalars().all())

    async def mark_sent(self, message: OutboundMessage) -> OutboundMessage:
        message.status = OutboundStatus.SENT
        message.sent_at = dt.datetime.now(dt.UTC)
        await self.session.flush()
        return message

    async def mark_failed(
        self, message: OutboundMessage, *, error: str, next_retry_at: dt.datetime | None
    ) -> OutboundMessage:
        message.attempt += 1
        message.last_error = error[:512]
        message.next_retry_at = next_retry_at
        message.status = (
            OutboundStatus.PENDING if next_retry_at is not None else OutboundStatus.FAILED
        )
        await self.session.flush()
        return message


class SqlDocumentRepository(DocumentRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, user_id: uuid.UUID, **fields: Any) -> Document:
        document = Document(user_id=user_id, **fields)
        self.session.add(document)
        await self.session.flush()
        return document

    async def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        return await self.session.get(Document, document_id)

    async def list_for_user(self, user_id: uuid.UUID, limit: int = 50) -> list[Document]:
        result = await self.session.execute(
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update_status(
        self, document: Document, status: DocumentStatus, **meta: Any
    ) -> Document:
        document.status = status
        if meta:
            document.doc_meta = {**(document.doc_meta or {}), **meta}
        await self.session.flush()
        return document


class SqlIntegrationRepository(IntegrationRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert(
        self,
        user_id: uuid.UUID,
        *,
        provider: IntegrationProvider,
        access_token: str,
        refresh_token: str | None = None,
        scopes: list[str] | None = None,
        expires_at: dt.datetime | None = None,
    ) -> IntegrationLink:
        result = await self.session.execute(
            select(IntegrationLink).where(
                IntegrationLink.user_id == user_id,
                IntegrationLink.provider == provider,
            )
        )
        link = result.scalar_one_or_none()
        if link is None:
            link = IntegrationLink(
                user_id=user_id,
                provider=provider,
                access_token=access_token,
                refresh_token=refresh_token,
                scopes=scopes,
                expires_at=expires_at,
            )
            self.session.add(link)
        else:
            link.access_token = access_token
            link.refresh_token = refresh_token
            link.scopes = scopes
            link.expires_at = expires_at
        await self.session.flush()
        return link

    async def get_by_provider(
        self, user_id: uuid.UUID, provider: IntegrationProvider
    ) -> IntegrationLink | None:
        result = await self.session.execute(
            select(IntegrationLink).where(
                IntegrationLink.user_id == user_id,
                IntegrationLink.provider == provider,
            )
        )
        return result.scalar_one_or_none()

    async def delete(self, link: IntegrationLink) -> None:
        await self.session.delete(link)
        await self.session.flush()


class SqlIngestLedgerRepository(IngestLedgerRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

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
        """Returns False when the update (or its message) was already ingested.

        Two uniqueness constraints protect idempotency:
          * ingested_updates.update_id             -> same update re-delivered
          * ingested_updates(chat_id, message_id)  -> same message re-delivered
        `ON CONFLICT DO NOTHING` without a target is accepted by both PostgreSQL
        and SQLite and fires on whichever constraint trips first.
        """
        values = {
            "update_id": update_id,
            "chat_id": chat_id,
            "message_id": message_id,
            "user_id": user_id,
            "source": source,
            "correlation_id": correlation_id,
        }
        dialect = self.session.sync_session.bind.dialect.name
        if dialect == "postgresql":
            stmt = pg_insert(IngestedUpdate).values(**values).on_conflict_do_nothing()
        else:
            stmt = sqlite_insert(IngestedUpdate).values(**values).on_conflict_do_nothing()
        result = await self.session.execute(stmt)
        await self.session.flush()
        return result.rowcount == 1
