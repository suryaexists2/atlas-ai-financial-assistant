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

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMGatewayError(RuntimeError):
    """Raised when the LLM provider cannot complete a request."""


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
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._http = http or httpx.AsyncClient(timeout=timeout_seconds)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 600,
        temperature: float = 0.3,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        attempt = 0
        while True:
            attempt += 1
            try:
                message, model, usage = await self._post(body)
                return self._parse_response(message, model, usage)
            except LLMGatewayTransientError:
                if attempt > self._max_retries:
                    raise
                await asyncio.sleep(self._backoff(attempt))

    async def _post(
        self, body: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None, dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = await self._http.post(self._base_url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise LLMGatewayError(f"LLM request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMGatewayError(f"LLM request failed: {exc}") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise LLMGatewayTransientError(
                f"LLM provider transient error {response.status_code}: {response.text[:200]}"
            )
        if response.status_code >= 400:
            raise LLMGatewayError(
                f"LLM provider error {response.status_code}: {response.text[:200]}"
            )

        payload = response.json()
        choice = (payload.get("choices") or [{}])[0]
        return (
            choice.get("message", {}),
            payload.get("model"),
            payload.get("usage", {}),
        )

    def _parse_response(
        self, message: dict[str, Any], model: str | None, usage: dict[str, Any]
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
            finish_reason=None,
        )

    def _backoff(self, attempt: int) -> float:
        return (2 ** (attempt - 1)) + random.uniform(0, 0.5)


__all__ = ["LLMGatewayError", "LLMGatewayTransientError", "OpenRouterGateway"]
