"""ORM entities (the persistence model).

These are the domain aggregates persisted via SQLAlchemy. Column types are
chosen to work identically on SQLite (dev/tests) and Postgres (production).
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.domain.enums import (
    AlertKind,
    ContentType,
    DocumentStatus,
    IntegrationProvider,
    JobStatus,
    MemoryStatus,
    MessageRole,
    OnboardingStatus,
    OutboundStatus,
)
from app.infrastructure.db.base import Base
from app.infrastructure.db.types import JSONType

_ENUM_ARGS = {"native_enum": False, "length": 32}


class _TimestampMixin:
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.UTC),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: dt.datetime.now(dt.UTC),
        onupdate=func.now(),
        nullable=False,
    )


class User(_TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    username: Mapped[str | None] = mapped_column(String(64))
    first_name: Mapped[str | None] = mapped_column(String(128))
    last_name: Mapped[str | None] = mapped_column(String(128))
    language_code: Mapped[str | None] = mapped_column(String(8))
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")

    profile: Mapped[UserProfile] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class UserProfile(_TimestampMixin, Base):
    __tablename__ = "user_profiles"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True
    )
    role: Mapped[str | None] = mapped_column(String(64))
    interests: Mapped[list[Any] | None] = mapped_column(JSONType)
    briefing_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    briefing_time: Mapped[str | None] = mapped_column(String(8))
    onboarding_status: Mapped[OnboardingStatus] = mapped_column(
        Enum(OnboardingStatus, **_ENUM_ARGS), default=OnboardingStatus.NOT_STARTED
    )
    onboarding_context: Mapped[dict[str, Any] | None] = mapped_column(JSONType)

    user: Mapped[User] = relationship(back_populates="profile")


class Conversation(_TimestampMixin, Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str | None] = mapped_column(String(255))

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(_TimestampMixin, Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole, **_ENUM_ARGS))
    content: Mapped[str | None] = mapped_column(Text)
    content_type: Mapped[ContentType] = mapped_column(
        Enum(ContentType, **_ENUM_ARGS), default=ContentType.TEXT
    )
    media_meta: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    correlation_id: Mapped[str | None] = mapped_column(String(36))

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class WatchlistItem(_TimestampMixin, Base):
    __tablename__ = "watchlist_items"
    __table_args__ = (UniqueConstraint("user_id", "symbol", name="uq_user_symbol"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str | None] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(128))
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Alert(_TimestampMixin, Base):
    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[AlertKind] = mapped_column(Enum(AlertKind, **_ENUM_ARGS))
    symbol: Mapped[str | None] = mapped_column(String(16))
    condition: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_fired_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))


class Document(_TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(512))
    mime_type: Mapped[str | None] = mapped_column(String(128))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, **_ENUM_ARGS), default=DocumentStatus.PENDING
    )
    doc_meta: Mapped[dict[str, Any] | None] = mapped_column(JSONType)


class Memory(_TimestampMixin, Base):
    __tablename__ = "memories"
    __table_args__ = (
        UniqueConstraint("user_id", "memory_key", "status", name="uq_user_key_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    memory_key: Mapped[str] = mapped_column(String(256), index=True)
    value: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    summary: Mapped[str | None] = mapped_column(String(512))
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    version: Mapped[int] = mapped_column(Integer, default=1)
    source_turn_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL")
    )
    first_seen_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_seen_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    status: Mapped[MemoryStatus] = mapped_column(
        Enum(MemoryStatus, **_ENUM_ARGS), default=MemoryStatus.ACTIVE
    )


class ScheduledJob(_TimestampMixin, Base):
    __tablename__ = "scheduled_jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    job_type: Mapped[str] = mapped_column(String(64), index=True)
    cron_expr: Mapped[str] = mapped_column(String(64))
    params: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_run_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    next_run_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), index=True)


class JobEvent(Base):
    __tablename__ = "job_events"
    __table_args__ = (UniqueConstraint("job_id", "run_key", name="uq_job_runkey"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("scheduled_jobs.id", ondelete="CASCADE"), index=True
    )
    run_key: Mapped[str] = mapped_column(String(64), index=True)
    scheduled_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    executed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, **_ENUM_ARGS), default=JobStatus.EXECUTED
    )


class OutboundMessage(Base):
    __tablename__ = "outbound_messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    chat_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[OutboundStatus] = mapped_column(
        Enum(OutboundStatus, **_ENUM_ARGS), default=OutboundStatus.PENDING
    )
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    next_retry_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    last_error: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    sent_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))


class IntegrationLink(_TimestampMixin, Base):
    __tablename__ = "integration_links"
    __table_args__ = (UniqueConstraint("user_id", "provider", name="uq_user_provider"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    provider: Mapped[IntegrationProvider] = mapped_column(Enum(IntegrationProvider, **_ENUM_ARGS))
    access_token: Mapped[str] = mapped_column(String(2048))
    refresh_token: Mapped[str | None] = mapped_column(String(2048))
    scopes: Mapped[list[str] | None] = mapped_column(JSONType)
    expires_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    linked_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class OAuthFlow(Base):
    """One-time, expiring server-side OAuth state (PKCE verifier bound to a user/chat)."""

    __tablename__ = "oauth_flows"

    state: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    code_verifier: Mapped[str] = mapped_column(String(128))
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    consumed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class IngestedUpdate(Base):
    """Temporary dedup ledger for raw Telegram updates (update_id / message_id)."""

    __tablename__ = "ingested_updates"
    __table_args__ = (UniqueConstraint("chat_id", "message_id", name="uq_chat_message"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    update_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    message_id: Mapped[int | None] = mapped_column(BigInteger)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    source: Mapped[str] = mapped_column(String(16))
    correlation_id: Mapped[str] = mapped_column(String(36), index=True)
    processed_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
