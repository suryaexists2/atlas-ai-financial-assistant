"""Telegram webhook endpoint.

Validates the `X-Telegram-Bot-Api-Secret-Token` header (constant-time) before
the payload touches any business logic. Requires `telegram_webhook_secret` to
be configured; when absent the endpoint refuses all traffic (safe default).
"""

from __future__ import annotations

import secrets
from typing import Any

from fastapi import APIRouter, Depends, Request

from app.core.config import Settings, get_settings
from app.core.context import RequestContext, new_correlation_id, push_context
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.interfaces.telegram.processor import UpdateProcessor

logger = get_logger(__name__)

router = APIRouter(tags=["telegram"])


def get_processor(request: Request) -> UpdateProcessor:
    processor: UpdateProcessor | None = getattr(request.app.state, "telegram_processor", None)
    if processor is None:
        raise AppError(
            "Telegram bot is not configured on this instance.",
            code="service_unavailable",
            status_code=503,
        )
    return processor


@router.post("/webhook/telegram")
async def telegram_webhook(
    request: Request,
    payload: dict[str, Any],
    processor: UpdateProcessor = Depends(get_processor),
    settings: Settings = Depends(get_settings),
) -> dict[str, bool]:
    expected = settings.telegram_webhook_secret
    provided = request.headers.get("X-Telegram-Bot-Api-Secret-Token") or ""
    if not expected or not secrets.compare_digest(provided, expected):
        raise AppError(
            "Webhook secret token missing or invalid.",
            code="forbidden",
            status_code=403,
        )

    correlation_id = request.headers.get("x-correlation-id") or new_correlation_id()
    push_context(
        RequestContext(
            correlation_id=correlation_id,
            telegram_chat_id=payload.get("message", {}).get("chat", {}).get("id"),
            source="webhook",
        )
    )
    await processor.process_update(
        payload,
        source="webhook",
        correlation_id=correlation_id,
        background_reply=True,
    )
    logger.info("webhook_handled", update_id=payload.get("update_id"))
    return {"ok": True}
