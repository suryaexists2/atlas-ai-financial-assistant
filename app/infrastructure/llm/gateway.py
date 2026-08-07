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
from typing import Any

import httpx

from app.application.agent.ports import LLMGateway, LLMResponse, LLMToolCall
from app.core.logging import get_logger

logger = get_logger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 600,
        temperature: float = 0.3,
    ) -> LLMResponse:
        base_body: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            base_body["tools"] = tools
            base_body["tool_choice"] = "auto"

        last_error: LLMGatewayError | None = None
        started = asyncio.get_running_loop().time()
        for model in self._models:
            body = dict(base_body)
            body["model"] = model
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
                    logger.info(
                        "llm_response_received",
                        model=model_used,
                        ms=round((asyncio.get_running_loop().time() - started) * 1000),
                        finish_reason=response.finish_reason,
                        n_tool_calls=len(response.tool_calls),
                        content_chars=len(response.content or ""),
                        usage=usage,
                    )
                    return response
                except LLMGatewayError as exc:
                    last_error = exc
                    if isinstance(exc, LLMGatewayTransientError) and attempt <= self._max_retries:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                    if exc.status_code in (401, 403):
                        raise
                    break
            logger.warning(
                "llm_fallback_model",
                model=model,
                error=str(last_error)[:200],
            )
        if len(self._models) == 1:
            raise last_error or LLMGatewayError("LLM provider request failed")
        raise LLMGatewayError(
            f"All {len(self._models)} LLM model(s) failed; last error: {last_error}"
        ) from last_error

    async def _post(
        self, body: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None, dict[str, Any], str | None]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
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

        if response.status_code == 429 or response.status_code >= 500:
            raise LLMGatewayTransientError(
                f"LLM provider transient error {response.status_code}: {response.text[:200]}",
                status_code=response.status_code,
            )
        if response.status_code >= 400:
            raise LLMGatewayError(
                f"LLM provider error {response.status_code}: {response.text[:200]}",
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
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
        )

    def _backoff(self, attempt: int) -> float:
        return (2 ** (attempt - 1)) + random.uniform(0, 0.5)


__all__ = ["LLMGatewayError", "LLMGatewayTransientError", "OpenRouterGateway"]
