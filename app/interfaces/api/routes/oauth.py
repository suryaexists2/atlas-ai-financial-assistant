"""Google OAuth endpoints.

Flow (all state is server-side and one-time):
  tool connect_google -> creates oauth_flows row -> reply carries a button to
  /oauth/google/start?state=...  -> redirect to Google consent ->
  /oauth/google/callback -> validates state (one-time, expiring), exchanges the
  code with PKCE, stores tokens, and enqueues a Telegram confirmation through
  the durable outbox. User/chat identity is read from the flow row only, never
  from callback query params.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.core.config import Settings, get_settings
from app.core.context import new_correlation_id
from app.core.logging import get_logger
from app.domain.enums import IntegrationProvider
from app.infrastructure.db.uow import UnitOfWork
from app.infrastructure.providers.google_oauth import (
    GoogleOAuthClient,
    GoogleOAuthError,
)

logger = get_logger(__name__)

router = APIRouter(tags=["oauth"])

_CLOSE_PAGE = (
    "<!doctype html><html><body style='font-family:sans-serif;text-align:center;padding-top:80px'>"
    "<h2>Connected to Atlas \u2713</h2><p>You can close this tab and return to Telegram.</p>"
    "</body></html>"
)

_FAIL_PAGE = (
    "<!doctype html><html><body style='font-family:sans-serif;text-align:center;padding-top:80px'>"
    "<h2>Connection failed</h2><p>The link was invalid or expired. Ask Atlas to "
    "connect Google again.</p>"
    "</body></html>"
)


def _oauth_client(settings: Settings) -> GoogleOAuthClient | None:
    if not settings.google_oauth_client_id or not settings.google_oauth_client_secret:
        return None
    if not settings.public_base_url:
        return None
    return GoogleOAuthClient(
        settings.google_oauth_client_id,
        settings.google_oauth_client_secret,
        redirect_uri=f"{settings.public_base_url.rstrip('/')}/oauth/google/callback",
        scopes=settings.google_oauth_scopes,
    )


async def _enqueue_confirmation(session_factory, chat_id: int, text: str) -> None:
    """Sends the connect result through the centralized durable outbox."""
    async with UnitOfWork(session_factory) as uow:
        await uow.outbox.enqueue(
            chat_id=chat_id,
            payload={
                "type": "text",
                "text": text,
                "correlation_id": new_correlation_id(),
            },
            priority=10,
        )


@router.get("/oauth/google/start", response_model=None)
async def oauth_start(
    request: Request,
    state: str = Query(..., min_length=20, max_length=80),
    settings: Settings = Depends(get_settings),
) -> RedirectResponse | HTMLResponse:
    client = _oauth_client(settings)
    if client is None:
        return HTMLResponse(
            "<h2>Google sign-in is not configured on this instance.</h2>", status_code=503
        )

    session_factory = request.app.state.session_factory
    async with UnitOfWork(session_factory) as uow:
        await uow.oauth_flows.delete_expired(dt.datetime.now(dt.UTC) - dt.timedelta(hours=1))
        # Look up without consuming: consumption happens only at the callback.
        flow = await uow.oauth_flows.get_by_state(state)
        if flow is None or flow.consumed:
            return HTMLResponse(_FAIL_PAGE, status_code=400)
        expires = flow.expires_at
        if expires is not None and expires.tzinfo is None:
            expires = expires.replace(tzinfo=dt.UTC)
        if expires is not None and expires <= dt.datetime.now(dt.UTC):
            return HTMLResponse(_FAIL_PAGE, status_code=400)
        auth_url = client.authorization_url(flow.state, flow.code_verifier)
    return RedirectResponse(auth_url)


@router.get("/oauth/google/callback", response_model=None)
async def oauth_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None, min_length=20, max_length=80),
    error: str | None = Query(default=None),
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    client = _oauth_client(settings)
    if client is None:
        return HTMLResponse(
            "<h2>Google sign-in is not configured on this instance.</h2>", status_code=503
        )

    session_factory = request.app.state.session_factory
    if error is not None or not code or not state:
        # Honest decline: tell the chat the connection was cancelled.
        await _notify_cancelled(session_factory, state)
        return HTMLResponse(_FAIL_PAGE, status_code=400)

    async with UnitOfWork(session_factory) as uow:
        flow = await uow.oauth_flows.consume(state)
        if flow is None:
            logger.info("oauth_state_rejected", state_prefix=state[:8])
            return HTMLResponse(_FAIL_PAGE, status_code=400)
        try:
            bundle = await client.exchange_code(code, flow.code_verifier)
        except GoogleOAuthError as exc:
            logger.warning("oauth_exchange_failed", kind=exc.kind, state_prefix=state[:8])
            return HTMLResponse(_FAIL_PAGE, status_code=400)
        expires_at = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=bundle.expires_in)
        for provider in (
            IntegrationProvider.GMAIL,
            IntegrationProvider.CALENDAR,
            IntegrationProvider.DRIVE,
        ):
            await uow.integrations.upsert(
                flow.user_id,
                provider=provider,
                access_token=bundle.access_token,
                refresh_token=bundle.refresh_token,
                scopes=bundle.scope,
                expires_at=expires_at,
            )
        chat_id = flow.chat_id
        await uow.commit()

    await _enqueue_confirmation(
        session_factory,
        chat_id,
        "Google connected ✓ You can now ask me to search your emails, check "
        'your calendar, or analyze files in Drive. Say "disconnect google" '
        "anytime to revoke access.",
    )
    logger.info("google_connected", chat_id=chat_id, providers=["gmail", "calendar", "drive"])
    return HTMLResponse(_CLOSE_PAGE)


async def _notify_cancelled(session_factory, state: str | None) -> None:
    if not state:
        return
    chat_id: int | None = None
    async with UnitOfWork(session_factory) as uow:
        flow = await uow.oauth_flows.consume(state)
        if flow is not None:
            chat_id = flow.chat_id
    if chat_id is not None:
        await _enqueue_confirmation(
            session_factory,
            chat_id,
            "Google connection was cancelled. You can ask me to connect Google again "
            "whenever you're ready.",
        )
