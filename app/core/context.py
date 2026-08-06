"""Request context and correlation-id propagation.

Every unit of work entering the system (webhook, polling update, scheduler job,
tool call, outbound delivery) carries a `RequestContext` with a correlation_id.
The context lives in a ContextVar so structlog and any async code can read it
without threading a parameter through every call.
"""

from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

_CORRELATION_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="unbound"
)


@dataclass
class RequestContext:
    correlation_id: str
    telegram_user_id: int | None = None
    telegram_chat_id: int | None = None
    user_id: str | None = None  # internal UUID, string form
    source: str | None = None  # webhook | polling | scheduler | outbound
    extra: dict[str, Any] = field(default_factory=dict)


def new_correlation_id() -> str:
    return uuid.uuid4().hex


def push_context(ctx: RequestContext) -> None:
    _CORRELATION_ID.set(ctx.correlation_id)
    structlog.contextvars.bind_contextvars(
        correlation_id=ctx.correlation_id,
        telegram_user_id=ctx.telegram_user_id,
        telegram_chat_id=ctx.telegram_chat_id,
        user_id=ctx.user_id,
        source=ctx.source,
    )


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()
    _CORRELATION_ID.set("unbound")


def current_correlation_id() -> str:
    return _CORRELATION_ID.get()


def current_context() -> RequestContext:
    values = structlog.contextvars.get_contextvars()
    return RequestContext(
        correlation_id=current_correlation_id(),
        telegram_user_id=values.get("telegram_user_id"),
        telegram_chat_id=values.get("telegram_chat_id"),
        user_id=values.get("user_id"),
        source=values.get("source"),
    )
