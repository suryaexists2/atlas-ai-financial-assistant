"""OpenRouter free-model discovery with capability filtering.

OpenRouter's `:free` route catalogue changes without notice (models retire and
appear constantly), so a static fallback chain alone is fragile. This registry
periodically fetches the public catalogue (no API key required) and keeps the
models that are actually compatible with how Atlas calls the LLM:

- chat-completion routes only: the request always carries `tools`, so models
  that cannot do tool calling are useless here
- free routes only ($0 prompt and completion pricing)
- large enough context window and completion cap for our prompt and output
- reasoning must be optional (mandatory reasoning burns the output budget)

Compatibility is re-checked on every refresh, so retirements drop out and new
models join automatically. Anything that slips past the filter and misbehaves
at runtime is handled by the gateway's skip-list, not here.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

_MODELS_URL = "https://openrouter.ai/api/v1/models"

# id/name fragments that mark models unsuitable for assistant turns
# (embedders, guardrails, code-only agents, audio/vision pipelines, ...).
_DENIED_FRAGMENTS = (
    "embed",
    "rerank",
    "guardrail",
    "moderation",
    "safety",
    "asr",
    "tts",
    "speech",
    "audio",
    "image",
    "vision",
    "search",
    "classifier",
    "drill",
    "sql",
    "code",
)


@dataclass(frozen=True)
class FreeModel:
    """A free-route model compatible with Atlas' tool-calling usage."""

    id: str
    context_length: int = 0
    max_completion_tokens: int | None = None
    # True when the request may send `reasoning: {"enabled": false}`.
    reasoning_disablable: bool = False


def _denied(model_id: str, name: str) -> bool:
    haystack = f"{model_id} {name}".lower()
    return any(fragment in haystack for fragment in _DENIED_FRAGMENTS)


def _num(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def filter_free_models(
    models: list[dict[str, Any]],
    *,
    min_context: int = 32_000,
    min_completion: int = 600,
) -> list[FreeModel]:
    """Return `:free` models that pass the capability/compatibility filter.

    Results are ordered by context length (largest first) so the most
    capable routes are tried first.
    """
    compatible: list[FreeModel] = []
    for raw in models:
        model_id = str(raw.get("id") or "")
        if not model_id.endswith(":free"):
            continue
        if _denied(model_id, str(raw.get("name") or "")):
            continue
        pricing = raw.get("pricing") or {}
        if str(pricing.get("prompt") or "0").strip("$") != "0":
            continue
        if str(pricing.get("completion") or "0").strip("$") != "0":
            continue
        parameters = [str(p).lower() for p in raw.get("supported_parameters") or []]
        if "tools" not in parameters:
            continue
        reasoning = raw.get("reasoning") or {}
        if reasoning.get("mandatory"):
            continue
        top = raw.get("top_provider") or {}
        context = _num(top.get("context_length")) or _num(raw.get("context_length")) or 0
        if context and context < min_context:
            continue
        max_completion = _num(top.get("max_completion_tokens"))
        if max_completion and max_completion < min_completion:
            continue
        compatible.append(
            FreeModel(
                id=model_id,
                context_length=context,
                max_completion_tokens=max_completion,
                reasoning_disablable="reasoning" in parameters,
            )
        )
    compatible.sort(key=lambda m: (m.context_length, m.id), reverse=True)
    return compatible


class FreeModelRegistry:
    """Cached view of the OpenRouter free-model catalogue.

    Safe to share across gateways: refreshes are serialized with a lock and
    failures keep the last known-good list so the bot never blocks on the
    catalogue.
    """

    def __init__(
        self,
        *,
        min_context: int = 32_000,
        min_completion: int = 600,
        refresh_seconds: int = 21_600,
        url: str = _MODELS_URL,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._min_context = min_context
        self._min_completion = min_completion
        self._refresh_seconds = refresh_seconds
        self._url = url
        self._http = http or httpx.AsyncClient(timeout=10.0)
        self._extras: list[FreeModel] = []
        self._last_refresh: float | None = None
        self._lock = asyncio.Lock()

    def extra_models(self) -> list[FreeModel]:
        return list(self._extras)

    async def refresh(self) -> bool:
        """Re-fetch the catalogue. Throttled to at most once per 30s."""
        async with self._lock:
            if self._last_refresh is not None and time.monotonic() - self._last_refresh < 30.0:
                return True
            try:
                response = await self._http.get(self._url)
                response.raise_for_status()
                payload = response.json()
                models = payload.get("data") if isinstance(payload, dict) else payload
                self._extras = filter_free_models(
                    models,
                    min_context=self._min_context,
                    min_completion=self._min_completion,
                )
                self._last_refresh = time.monotonic()
                logger.info(
                    "llm_free_models_refreshed",
                    n=len(self._extras),
                    models=[m.id for m in self._extras],
                )
                return True
            except Exception as exc:  # noqa: BLE001 - registry must never raise
                logger.warning("llm_free_models_refresh_failed", error=str(exc)[:200])
                return False

    async def ensure_fresh(self) -> list[FreeModel]:
        """Refresh when stale (or never fetched) and return the current list."""
        stale = self._last_refresh is None
        if not stale:
            stale = time.monotonic() - self._last_refresh >= self._refresh_seconds
        if stale:
            await self.refresh()
        return self.extra_models()


_REGISTRY: FreeModelRegistry | None = None


def get_registry(settings=None) -> FreeModelRegistry:
    """Process-wide registry singleton (one catalogue fetch per refresh period)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = FreeModelRegistry(
            min_context=settings.llm_free_min_context if settings is not None else 32_000,
            min_completion=settings.llm_free_min_completion if settings is not None else 600,
            refresh_seconds=settings.llm_models_refresh_seconds if settings is not None else 21_600,
        )
    return _REGISTRY


__all__ = ["FreeModel", "FreeModelRegistry", "filter_free_models", "get_registry"]
