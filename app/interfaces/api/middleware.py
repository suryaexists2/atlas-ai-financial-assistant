"""ASGI middleware that seeds request context and correlation ids."""

from typing import Any

from starlette.datastructures import Headers

from app.core.context import (
    RequestContext,
    clear_context,
    new_correlation_id,
    push_context,
)


class CorrelationMiddleware:
    """Generates/reuses a correlation id and exposes it via the RequestContext."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        correlation_id = headers.get("x-correlation-id") or new_correlation_id()
        push_context(RequestContext(correlation_id=correlation_id))
        try:
            await self.app(scope, receive, send)
        finally:
            clear_context()
