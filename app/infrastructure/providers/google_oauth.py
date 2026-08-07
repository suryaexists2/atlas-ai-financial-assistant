"""Google OAuth 2.0 client (PKCE S256) for Gmail / Calendar / Drive.

Pure HTTP via httpx — no google-api-python-client dependency. Tokens and
client secrets are never logged; error messages are scrubbed of token values.

Flow:
  build_authorization_url(state, code_verifier)  -> accounts.google.com consent
  exchange_code(code, code_verifier)             -> tokens + email (offline)
  refresh_access_token(refresh_token)            -> new access token (offline)
  revoke(refresh_token)                          -> invalidates the grant
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

_TOKEN_URL = "https://oauth2.googleapis.com/token"
_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_REVOKE_URL = "https://oauth2.googleapis.com/revoke"
_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

_UA = "Mozilla/5.0 (compatible; AtlasAI/0.1; +https://atlas-bot-peop.onrender.com)"


class GoogleOAuthError(RuntimeError):
    """Raised for token/authorization failures. `kind` drives friendly replies."""

    def __init__(self, message: str, *, kind: str = "oauth") -> None:
        super().__init__(message)
        self.kind = kind


class GoogleTokenExpiredError(RuntimeError):
    """Access token rejected with 401/403 — caller should refresh and retry once."""


class GoogleApiError(RuntimeError):
    """Raised for Gmail/Calendar/Drive API errors (never includes token data)."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class TokenBundle:
    access_token: str
    refresh_token: str | None
    expires_in: int
    scope: list[str]
    email: str | None = None


def generate_state() -> str:
    return secrets.token_urlsafe(32)


def generate_verifier() -> str:
    return secrets.token_urlsafe(48)


def _challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


class GoogleOAuthClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scopes: list[str],
        *,
        timeout_seconds: float = 20.0,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri
        self._scopes = list(scopes)
        self._http = http or httpx.AsyncClient(timeout=timeout_seconds)
        self._refresh_locks: dict[str, asyncio.Lock] = {}

    @property
    def configured(self) -> bool:
        return bool(self._client_id and self._client_secret)

    def authorization_url(self, state: str, code_verifier: str) -> str:
        params = {
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri,
            "scope": " ".join(self._scopes),
            "state": state,
            "code_challenge": _challenge(code_verifier),
            "code_challenge_method": "S256",
            "access_type": "offline",
            "prompt": "consent",
        }
        return f"{_AUTH_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"

    async def exchange_code(self, code: str, code_verifier: str) -> TokenBundle:
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self._redirect_uri,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "code_verifier": code_verifier,
        }
        data = await self._post_form(_TOKEN_URL, form)
        return TokenBundle(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            expires_in=int(data.get("expires_in", 3600)),
            scope=(data.get("scope", "").split() or self._scopes),
        )

    async def refresh_access_token(self, refresh_token: str) -> TokenBundle:
        form = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        data = await self._post_form(_TOKEN_URL, form)
        return TokenBundle(
            access_token=data["access_token"],
            refresh_token=refresh_token,
            expires_in=int(data.get("expires_in", 3600)),
            scope=(data.get("scope", "").split() or self._scopes),
        )

    async def userinfo(self, access_token: str) -> dict[str, Any]:
        try:
            response = await self._http.get(
                _USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}", "User-Agent": _UA},
            )
        except httpx.HTTPError as exc:
            raise GoogleOAuthError("Google userinfo is unreachable", kind="network") from exc
        if response.status_code == 401:
            raise GoogleTokenExpiredError()
        if response.status_code != 200:
            raise GoogleOAuthError("Google userinfo failed", kind="oauth")
        return response.json()

    async def revoke(self, token: str) -> None:
        """Revokes a refresh (or access) token; idempotent."""
        form = {"token": token}
        try:
            response = await self._http.post(
                _REVOKE_URL,
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        except httpx.HTTPError as exc:
            raise GoogleOAuthError("Google revoke endpoint is unreachable", kind="network") from exc
        # 200 or 400 with invalid_token are both acceptable outcomes.
        if response.status_code not in (200, 400):
            raise GoogleOAuthError("Google revoke failed", kind="oauth")

    def lock_for(self, user_id: uuid.UUID) -> asyncio.Lock:
        return self._refresh_locks.setdefault(str(user_id), asyncio.Lock())

    async def _post_form(self, url: str, form: dict[str, str]) -> dict[str, Any]:
        try:
            response = await self._http.post(
                url,
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": _UA},
            )
        except httpx.HTTPError as exc:
            raise GoogleOAuthError("Google token endpoint is unreachable", kind="network") from exc
        if response.status_code != 200:
            body = response.text[:300] if response.text else ""
            kind = "invalid_grant" if "invalid_grant" in body else "oauth"
            raise GoogleOAuthError("Google rejected the token request", kind=kind)
        return response.json()

    async def aclose(self) -> None:
        await self._http.aclose()


__all__ = [
    "GoogleOAuthClient",
    "GoogleOAuthError",
    "GoogleTokenExpiredError",
    "GoogleApiError",
    "TokenBundle",
    "generate_state",
    "generate_verifier",
]
