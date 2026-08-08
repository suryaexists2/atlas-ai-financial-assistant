"""OpenRouter chat-completion gateway.

Implements the `LLMGateway` port against the OpenAI-compatible endpoint at
`https://openrouter.ai/api/v1/chat/completions`. Supports tool calling via the
standard `tools`/`tool_calls` shape, with bounded retries on transient errors
(429/5xx) and a hard timeout so agent turns never hang forever.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

import httpx

from app.application.agent.ports import LLMGateway, LLMResponse, LLMToolCall
from app.core.logging import get_logger
from app.infrastructure.llm.keys import GroqKeyPool
from app.infrastructure.llm.models_registry import FreeModelRegistry

logger = get_logger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMGatewayError(RuntimeError):
    """Raised when the LLM provider cannot complete a request."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMGatewayTransientError(LLMGatewayError):
    """Provider returned 429 or 5xx — safe to retry."""


class OpenRouterGateway(LLMGateway):
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = _OPENROUTER_URL,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        fallback_models: list[str] | None = None,
        http: httpx.AsyncClient | None = None,
        skip_seconds: int = 600,
        rate_limit_skip_seconds: int = 60,
        registry: FreeModelRegistry | None = None,
        key_pool: GroqKeyPool | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._http = http or httpx.AsyncClient(timeout=timeout_seconds)
        self._models = [model]
        for extra in fallback_models or []:
            if extra and extra != model and extra not in self._models:
                self._models.append(extra)
        self._skip_seconds = skip_seconds
        # 429/rate limits reset fast (seconds), auth/404 problems persist
        # (minutes) — a short 429 penalty keeps the primary engine in play.
        self._rate_limit_skip_seconds = min(rate_limit_skip_seconds, skip_seconds)
        self._registry = registry
        # model -> expiry timestamp (monotonic); skipped models are not tried.
        self._skip_until: dict[str, float] = {}
        # registry-managed models that accept `reasoning: {"enabled": false}`.
        self._reasoning_off: dict[str, dict[str, Any]] = {}
        # per-model extra request body (e.g. Groq qwen needs reasoning off and
        # a parsed reasoning format for tool calling; gpt-oss must not get
        # either parameter).
        self._extra_body: dict[str, dict[str, Any]] = {}
        # Multi-key failover pool (Groq). None keeps the single-key behavior.
        self._key_pool = key_pool
        # Retry pause for tokens-per-minute (TPM) 413s from Groq.
        self._tpm_retry_seconds = 45.0

    async def _sync_registry(self) -> None:
        if self._registry is None:
            return
        try:
            await self._registry.ensure_fresh()
        except Exception:  # noqa: BLE001 - the chain must work without the catalogue
            logger.warning("llm_registry_lookup_failed", exc_info=True)
        for extra in self._registry.extra_models():
            if extra.id not in self._models:
                self._models.append(extra.id)
            if extra.reasoning_disablable:
                self._reasoning_off[extra.id] = {"enabled": False}

    def _skip(self, model: str, status_code: int | None = None) -> None:
        duration = (
            self._rate_limit_skip_seconds if status_code == 429 else self._skip_seconds
        )
        self._skip_until[model] = time.monotonic() + duration

    def _chain(self) -> list[str]:
        now = time.monotonic()
        return [m for m in self._models if now >= self._skip_until.get(m, 0.0)]

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 600,
        temperature: float = 0.3,
    ) -> LLMResponse:
        await self._sync_registry()
        chain = self._chain()
        if not chain:
            raise LLMGatewayError(
                "All LLM models are temporarily skipped after recent failures; try again shortly"
            )
        base_body: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            base_body["tools"] = tools
            base_body["tool_choice"] = "auto"

        last_error: LLMGatewayError | None = None
        last_empty: LLMResponse | None = None
        self._model_errors: dict[str, str] = {}
        started = asyncio.get_running_loop().time()
        for model in chain:
            body = dict(base_body)
            body["model"] = model
            if model in self._reasoning_off:
                body["reasoning"] = self._reasoning_off[model]
            body.update(self._extra_body.get(model, {}))
            attempt = 0
            while True:
                attempt += 1
                logger.info(
                    "llm_request_sent",
                    model=model,
                    attempt=attempt,
                    n_messages=len(messages),
                    n_tools=len(tools or []),
                )
                try:
                    message, model_used, usage, finish_reason = await self._post(body)
                    response = self._parse_response(message, model_used, usage, finish_reason)
                    outcome = {
                        "model": model_used,
                        "ms": round((asyncio.get_running_loop().time() - started) * 1000),
                        "finish_reason": finish_reason,
                        "n_tool_calls": len(response.tool_calls),
                        "content_chars": len(response.content or ""),
                    }
                    if not (response.content or "").strip() and not response.tool_calls:
                        # Success code, but nothing usable — keep the only failure
                        # evidence for the caller and try the next model.
                        logger.warning("llm_empty_response", **outcome)
                        last_empty = response
                        self._skip(model)
                        break
                    logger.info("llm_response_received", **outcome, usage=usage)
                    return response
                except LLMGatewayError as exc:
                    last_error = exc
                    self._model_errors[model] = str(exc)[:400]
                    if isinstance(exc, LLMGatewayTransientError) and attempt <= self._max_retries:
                        # TPM windows roll over in ~60s; the generic backoff is
                        # seconds, so space those retries out properly.
                        wait = (
                            self._tpm_retry_seconds
                            if exc.status_code == 413
                            else self._backoff(attempt)
                        )
                        await asyncio.sleep(wait)
                        continue
                    if exc.status_code in (401, 403):
                        raise
                    if exc.status_code is not None:
                        self._skip(model, exc.status_code)
                    break
            logger.warning(
                "llm_fallback_model",
                model=model,
                error=str(last_error or last_empty)[:200],
            )
        if last_empty is not None:
            raise LLMGatewayError(
                "LLM returned empty content for every model (no text, no tool calls); "
                f"last finish_reason={last_empty.finish_reason} model={last_empty.model}"
            )
        if len(chain) == 1:
            raise last_error or LLMGatewayError("LLM provider request failed")
        raise LLMGatewayError(
            f"All {len(chain)} LLM model(s) failed; "
            + "; ".join(f"{m}={err}" for m, err in self._model_errors.items())[:1500]
        ) from last_error

    async def _post(
        self, body: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None, dict[str, Any], str | None]:
        key: str | None = self._key_pool.current() if self._key_pool else self._api_key
        if key is None:
            raise LLMGatewayError(
                "All Groq API keys are exhausted (daily token caps); try again after the reset"
            )
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        try:
            t0 = asyncio.get_running_loop().time()
            response = await self._http.post(self._base_url, json=body, headers=headers)
            logger.info(
                "llm_http_completed",
                status=response.status_code,
                ms=round((asyncio.get_running_loop().time() - t0) * 1000),
                body_chars=len(response.text),
            )
        except httpx.TimeoutException as exc:
            raise LLMGatewayError(f"LLM request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMGatewayError(f"LLM request failed: {exc}") from exc

        if (
            response.status_code == 429
            and key is not None
            and self._key_pool is not None
            and len(self._key_pool.keys) > 1
        ):
            # Daily caps are per account: only fail over when a spare key
            # exists — a single key still retries like any transient error.
            self._key_pool.mark_exhausted(key)
        if response.status_code == 429 or response.status_code >= 500:
            raise LLMGatewayTransientError(
                f"LLM provider transient error {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )
        if response.status_code == 413:
            # Groq returns 413 when the request bursts the model's tokens-per-
            # minute window (body mentions TPM); the window rolls over shortly,
            # so it retries like a rate limit. Other 413s (context length) are
            # hard failures that retrying cannot help.
            if "tokens per minute" in response.text.lower():
                raise LLMGatewayTransientError(
                    f"LLM provider transient error 413 (TPM window): {response.text[:200]}",
                    status_code=413,
                )
            raise LLMGatewayError(
                f"LLM provider error 413: {response.text[:400]}",
                status_code=413,
            )
        if response.status_code >= 400:
            raise LLMGatewayError(
                f"LLM provider error {response.status_code}: {response.text[:400]}",
                status_code=response.status_code,
            )

        payload = response.json()
        choice = (payload.get("choices") or [{}])[0]
        return (
            choice.get("message", {}),
            payload.get("model"),
            payload.get("usage", {}),
            choice.get("finish_reason"),
        )

    def _parse_response(
        self,
        message: dict[str, Any],
        model: str | None,
        usage: dict[str, Any],
        finish_reason: str | None = None,
    ) -> LLMResponse:
        tool_calls: list[LLMToolCall] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function", {})
            arguments = function.get("arguments") or "{}"
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {}
            tool_calls.append(
                LLMToolCall(
                    id=raw.get("id", ""),
                    name=function.get("name", ""),
                    arguments=parsed,
                )
            )
        content = message.get("content")
        raw = {
            key: repr(message.get(key))[:200]
            for key in ("content", "tool_calls", "refusal", "reasoning", "reasoning_content")
            if message.get(key) is not None
        }
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
            raw=raw,
        )

    def _backoff(self, attempt: int) -> float:
        return (2 ** (attempt - 1)) + random.uniform(0, 0.5)


class GroqGateway(OpenRouterGateway):
    """Free chat completions via Groq's OpenAI-compatible API.

    Same protocol and failover logic as the OpenRouter gateway, but pointed at
    Groq (gpt-oss / qwen routes). No free-model registry — the Groq catalogue
    is stable. Qwen models get `reasoning_effort: "none"` (their thinking eats
    the reply budget and is not the deliverable) plus `reasoning_format:
    "parsed"`, which Groq requires for tool calling; gpt-oss models must not
    receive either parameter, so they are applied per-model. When a `key_pool`
    is provided, rate-limited (429) keys are parked and the next key is used
    for the retry — one key per request, switching only on exhaustion.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        fallback_models: list[str] | None = None,
        http: httpx.AsyncClient | None = None,
        skip_seconds: int = 600,
        rate_limit_skip_seconds: int = 60,
        key_pool: GroqKeyPool | None = None,
    ) -> None:
        super().__init__(
            api_key,
            model,
            base_url=_GROQ_URL,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            fallback_models=fallback_models,
            http=http,
            skip_seconds=skip_seconds,
            rate_limit_skip_seconds=rate_limit_skip_seconds,
            registry=None,
            key_pool=key_pool or GroqKeyPool([api_key]),
        )
        for candidate in self._models:
            if "qwen" in candidate:
                self._extra_body[candidate] = {
                    "reasoning_effort": "none",
                    "reasoning_format": "parsed",
                }


class FailoverGateway(LLMGateway):
    """Tries the primary gateway, then transparently falls back to the backup.

    Keeps the agent alive when one provider rate-limits or runs out of credits
    (e.g. Groq free tier -> OpenRouter free chain). Only `LLMGatewayError`
    failures trigger the switch; success on either side is returned as-is.
    """

    def __init__(self, primary: LLMGateway, backup: LLMGateway) -> None:
        self._primary = primary
        self._backup = backup

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 600,
        temperature: float = 0.3,
    ) -> LLMResponse:
        try:
            return await self._primary.complete(
                messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except LLMGatewayError as primary_error:
            logger.warning("llm_provider_failover", error=str(primary_error)[:200])
            try:
                return await self._backup.complete(
                    messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except LLMGatewayError as backup_error:
                # Keep the primary's reason visible: two independent providers
                # failing usually hides a config problem (e.g. bad key list),
                # which the caller otherwise never sees.
                raise LLMGatewayError(
                    f"{backup_error}; primary provider: {primary_error}"
                ) from backup_error


__all__ = [
    "LLMGatewayError",
    "LLMGatewayTransientError",
    "OpenRouterGateway",
    "GroqGateway",
    "FailoverGateway",
]
