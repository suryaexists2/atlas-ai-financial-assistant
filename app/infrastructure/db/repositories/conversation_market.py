"""SQLAlchemy implementations of conversation/message and market-data repositories."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import Alert, Conversation, Message, WatchlistItem
from app.domain.enums import ContentType, MessageRole
from app.domain.repositories import (
    AlertRepository,
    ConversationRepository,
    WatchlistRepository,
)


class SqlConversationRepository(ConversationRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, user_id: uuid.UUID, *, title: str | None = None) -> Conversation:
        conversation = Conversation(user_id=user_id, title=title)
        self.session.add(conversation)
        await self.session.flush()
        return conversation

    async def get_by_id(self, conversation_id: uuid.UUID) -> Conversation | None:
        return await self.session.get(Conversation, conversation_id)

    async def list_for_user(self, user_id: uuid.UUID, limit: int = 20) -> list[Conversation]:
        result = await self.session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def add_message(
        self,
        conversation_id: uuid.UUID,
        *,
        role: MessageRole,
        content: str | None,
        content_type: Any = None,
        media_meta: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            content_type=content_type or ContentType.TEXT,
            media_meta=media_meta,
            correlation_id=correlation_id,
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def list_messages(self, conversation_id: uuid.UUID, limit: int = 50) -> list[Message]:
        result = await self.session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())


class SqlWatchlistRepository(WatchlistRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self, user_id: uuid.UUID, *, symbol: str, name: str | None, sector: str | None
    ) -> WatchlistItem:
        item = WatchlistItem(user_id=user_id, symbol=symbol, name=name, sector=sector)
        self.session.add(item)
        await self.session.flush()
        return item

    async def get_by_symbol(self, user_id: uuid.UUID, symbol: str) -> WatchlistItem | None:
        result = await self.session.execute(
            select(WatchlistItem).where(
                WatchlistItem.user_id == user_id,
                WatchlistItem.symbol == symbol.upper(),
                WatchlistItem.active.is_(True),
            )
        )
        return result.scalar_one_or_none()

    async def list_active(self, user_id: uuid.UUID) -> list[WatchlistItem]:
        result = await self.session.execute(
            select(WatchlistItem)
            .where(WatchlistItem.user_id == user_id, WatchlistItem.active.is_(True))
            .order_by(WatchlistItem.created_at.asc())
        )
        return list(result.scalars().all())

    async def deactivate(self, item: WatchlistItem) -> WatchlistItem:
        item.active = False
        await self.session.flush()
        return item


class SqlAlertRepository(AlertRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, user_id: uuid.UUID, **fields: Any) -> Alert:
        alert = Alert(user_id=user_id, **fields)
        self.session.add(alert)
        await self.session.flush()
        return alert

    async def get_by_id(self, alert_id: uuid.UUID) -> Alert | None:
        return await self.session.get(Alert, alert_id)

    async def list_enabled(self, user_id: uuid.UUID | None = None) -> list[Alert]:
        query = select(Alert).where(Alert.enabled.is_(True))
        if user_id is not None:
            query = query.where(Alert.user_id == user_id)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update(self, alert: Alert, **fields: Any) -> Alert:
        for key, value in fields.items():
            setattr(alert, key, value)
        await self.session.flush()
        return alert

    async def delete(self, alert: Alert) -> None:
        await self.session.delete(alert)
        await self.session.flush()
