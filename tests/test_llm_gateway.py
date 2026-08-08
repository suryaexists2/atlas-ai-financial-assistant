"""OpenRouter gateway tests: payload shape, tool-call parsing, retries, errors."""

import json
import time

import httpx
import pytest

from app.application.agent.ports import LLMResponse
from app.infrastructure.llm.gateway import (
    FailoverGateway,
    GroqGateway,
    LLMGatewayError,
    LLMGatewayTransientError,
    OpenRouterGateway,
)
from app.infrastructure.llm.keys import GroqKeyPool
from app.infrastructure.llm.models_registry import FreeModel, FreeModelRegistry


def _client_with(handler) -> OpenRouterGateway:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenRouterGateway("test-key", "model-x", max_retries=1, http=http)


@pytest.mark.asyncio
async def test_complete_plain_text():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "model-x"
        assert body["messages"][0]["role"] == "system"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "hello world"}}],
                "model": "model-x",
                "usage": {"total_tokens": 42},
            },
        )

    gateway = _client_with(handler)
    response = await gateway.complete(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    )
    assert response.content == "hello world"
    assert response.tool_calls == []
    assert response.usage == {"total_tokens": 42}


@pytest.mark.asyncio
async def test_complete_tool_call_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "tools" in body
        assert body["tool_choice"] == "auto"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_market_quote",
                                        "arguments": json.dumps({"symbol": "AAPL"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    gateway = _client_with(handler)
    response = await gateway.complete(
        [{"role": "user", "content": "quote aapl"}], tools=[{"type": "function"}]
    )
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "get_market_quote"
    assert response.tool_calls[0].arguments == {"symbol": "AAPL"}
    assert response.content is None


@pytest.mark.asyncio
async def test_bad_tool_arguments_are_tolerated():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c",
                                    "function": {"name": "x", "arguments": "not-json{"},
                                }
                            ]
                        }
                    }
                ]
            },
        )

    gateway = _client_with(handler)
    response = await gateway.complete([{"role": "user", "content": "hi"}])
    assert response.tool_calls[0].arguments == {}


@pytest.mark.asyncio
async def test_transient_error_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    gateway = _client_with(handler)
    response = await gateway.complete([{"role": "user", "content": "hi"}])
    assert response.content == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_transient_error_exhausts_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    gateway = _client_with(handler)
    with pytest.raises(LLMGatewayTransientError):
        await gateway.complete([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_http_error_maps_to_gateway_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    gateway = _client_with(handler)
    with pytest.raises(LLMGatewayError):
        await gateway.complete([{"role": "user", "content": "hi"}])


def _gateway_with(
    handler, *, model="model-x", fallbacks=None, max_retries=1, registry=None, skip_seconds=600
):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenRouterGateway(
        "test-key",
        model,
        max_retries=max_retries,
        fallback_models=fallbacks,
        http=http,
        registry=registry,
        skip_seconds=skip_seconds,
    )


@pytest.mark.asyncio
async def test_fallback_model_used_when_primary_route_fails():
    requested_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested_models.append(body["model"])
        if body["model"] == "model-x":
            return httpx.Response(404, text="model not found")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "from fallback"}}],
                "model": body["model"],
            },
        )

    gateway = _gateway_with(handler, fallbacks=["fallback-a", "fallback-a", "model-x"])
    response = await gateway.complete([{"role": "user", "content": "hi"}])
    assert response.content == "from fallback"
    assert requested_models == ["model-x", "fallback-a"]


@pytest.mark.asyncio
async def test_auth_error_does_not_rotate():
    requested_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested_models.append(body["model"])
        return httpx.Response(401, text="invalid api key")

    gateway = _gateway_with(handler, fallbacks=["fallback-a"])
    with pytest.raises(LLMGatewayError):
        await gateway.complete([{"role": "user", "content": "hi"}])
    assert requested_models == ["model-x"]


@pytest.mark.asyncio
async def test_all_models_failed_raises_combined_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="nope")

    gateway = _gateway_with(handler, fallbacks=["fallback-a", "fallback-b"])
    with pytest.raises(LLMGatewayError) as exc_info:
        await gateway.complete([{"role": "user", "content": "hi"}])
    assert "All 3 LLM model(s) failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_transient_error_rotates_after_retries_exhausted():
    requested_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested_models.append(body["model"])
        if body["model"] == "model-x":
            return httpx.Response(503, text="down")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "recovered"}}],
                "model": body["model"],
            },
        )

    gateway = _gateway_with(handler, fallbacks=["fallback-a"], max_retries=2)
    response = await gateway.complete([{"role": "user", "content": "hi"}])
    assert response.content == "recovered"
    assert requested_models == ["model-x", "model-x", "model-x", "fallback-a"]


@pytest.mark.asyncio
async def test_empty_response_rotates_to_next_model():
    requested_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested_models.append(body["model"])
        if body["model"] == "model-x":
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "real answer"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    gateway = _gateway_with(handler, fallbacks=["fallback-a"])
    response = await gateway.complete([{"role": "user", "content": "hi"}])
    assert response.content == "real answer"
    assert requested_models == ["model-x", "fallback-a"]


@pytest.mark.asyncio
async def test_all_models_empty_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "   "}, "finish_reason": "stop"}
                ]
            },
        )

    gateway = _gateway_with(handler, fallbacks=["fallback-a", "fallback-b"])
    with pytest.raises(LLMGatewayError) as exc_info:
        await gateway.complete([{"role": "user", "content": "hi"}])
    assert "empty content for every model" in str(exc_info.value)


@pytest.mark.asyncio
async def test_402_failover_to_free_backup_and_skipped_next_turn():
    requested_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested_models.append(body["model"])
        if body["model"] in ("model-x", "paid-a"):
            return httpx.Response(402, text="insufficient credits; can only afford 385 tokens")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "free reply"}}],
                "model": body["model"],
            },
        )

    gateway = _gateway_with(handler, fallbacks=["paid-a", "free-a"])
    first = await gateway.complete([{"role": "user", "content": "hi"}])
    assert first.content == "free reply"
    assert requested_models == ["model-x", "paid-a", "free-a"]

    second = await gateway.complete([{"role": "user", "content": "hi"}])
    assert second.content == "free reply"
    assert requested_models == ["model-x", "paid-a", "free-a", "free-a"]


@pytest.mark.asyncio
async def test_skipped_model_not_requested_again_even_if_healthy():
    requested_models: list[str] = []
    state = {"down": True}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested_models.append(body["model"])
        if state["down"] and body["model"] == "model-x":
            return httpx.Response(404, text="model not found")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "model": body["model"],
            },
        )

    gateway = _gateway_with(handler, fallbacks=["fallback-a"])
    await gateway.complete([{"role": "user", "content": "hi"}])
    assert requested_models == ["model-x", "fallback-a"]

    state["down"] = False
    await gateway.complete([{"role": "user", "content": "hi"}])
    assert requested_models == ["model-x", "fallback-a", "fallback-a"]


@pytest.mark.asyncio
async def test_all_models_skipped_raises_clean_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="nope")

    gateway = _gateway_with(handler, fallbacks=["fallback-a"])
    with pytest.raises(LLMGatewayError):
        await gateway.complete([{"role": "user", "content": "hi"}])
    with pytest.raises(LLMGatewayError) as exc_info:
        await gateway.complete([{"role": "user", "content": "hi"}])
    assert "temporarily skipped" in str(exc_info.value)


@pytest.mark.asyncio
async def test_registry_extras_appended_and_reasoning_disabled():
    requested_bodies: list[dict] = []
    registry = FreeModelRegistry()
    registry._last_refresh = time.monotonic()
    registry._extras = [
        FreeModel(id="registry-free:free", context_length=100000, reasoning_disablable=True)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested_bodies.append(body)
        if body["model"] in ("model-x", "cfg-free:free"):
            return httpx.Response(404, text="gone")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "model": body["model"],
            },
        )

    gateway = _gateway_with(handler, fallbacks=["cfg-free:free"], registry=registry)
    response = await gateway.complete([{"role": "user", "content": "hi"}])
    assert response.content == "done"
    expected = ["model-x", "cfg-free:free", "registry-free:free"]
    assert [b["model"] for b in requested_bodies] == expected
    assert requested_bodies[-1]["reasoning"] == {"enabled": False}


@pytest.mark.asyncio
async def test_skip_expiry_allows_model_to_be_tried_again():
    requested_models: list[str] = []
    state = {"down": True}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested_models.append(body["model"])
        if state["down"] and body["model"] == "model-x":
            return httpx.Response(402, text="no credits")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "model": body["model"],
            },
        )

    gateway = _gateway_with(handler, fallbacks=["fallback-a"], skip_seconds=0)
    first = await gateway.complete([{"role": "user", "content": "hi"}])
    assert first.content == "ok"
    assert requested_models == ["model-x", "fallback-a"]
    state["down"] = False
    second = await gateway.complete([{"role": "user", "content": "hi"}])
    assert second.content == "ok"
    assert requested_models == ["model-x", "fallback-a", "model-x"]


class _FakeGateway:
    """Stub LLMGateway for FailoverGateway tests."""

    def __init__(self, *, content="ok", error=None, used=None):
        self._content = content
        self._error = error
        self._used = used or []
        self.calls = 0

    async def complete(self, messages, *, tools=None, max_tokens=600, temperature=0.3):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return LLMResponse(content=self._content, model="fake")


@pytest.mark.asyncio
async def test_groq_gateway_uses_groq_endpoint():
    requested = {"url": None, "model": None}

    def handler(request: httpx.Request) -> httpx.Response:
        requested["url"] = str(request.url)
        body = json.loads(request.content)
        requested["model"] = body["model"]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "groq reply"}}],
                "model": body["model"],
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = GroqGateway(
        "groq-key",
        "llama-3.3-70b-versatile",
        fallback_models=["llama-3.1-8b-instant"],
        http=http,
    )
    response = await gateway.complete([{"role": "user", "content": "hi"}])
    assert response.content == "groq reply"
    assert requested["url"].startswith("https://api.groq.com/openai/v1/chat/completions")
    assert requested["model"] == "llama-3.3-70b-versatile"


@pytest.mark.asyncio
async def test_groq_gateway_uses_qwen_fallback_and_tunes_reasoning_per_model():
    """When gpt-oss rate-limits, the Groq chain falls back to qwen3.6-27b. The
    qwen model must get `reasoning_effort: none` + `reasoning_format: parsed`
    (required for tool calling) while gpt-oss keeps its vanilla body."""
    requested: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested.append(body)
        if body["model"] == "openai/gpt-oss-120b":
            return httpx.Response(429, text="tokens per day exceeded")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "via qwen"}}],
                "model": body["model"],
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = GroqGateway(
        "groq-key",
        "openai/gpt-oss-120b",
        max_retries=1,
        fallback_models=["qwen/qwen3.6-27b"],
        http=http,
    )
    response = await gateway.complete(
        [{"role": "user", "content": "hi"}], tools=[{"type": "function"}]
    )
    assert response.content == "via qwen"

    gpt_oss = next(b for b in requested if b["model"] == "openai/gpt-oss-120b")
    qwen = next(b for b in requested if b["model"] == "qwen/qwen3.6-27b")
    assert "reasoning_effort" not in gpt_oss
    assert "reasoning_format" not in gpt_oss
    assert qwen["reasoning_effort"] == "none"
    assert qwen["reasoning_format"] == "parsed"
    assert "tools" in qwen


@pytest.mark.asyncio
async def test_groq_gateway_switches_key_on_ratelimit():
    """One key per request: the request is retried with the next key only after
    a 429, and the pool stays on the healthy key afterwards."""
    pool = GroqKeyPool(["key-a", "key-b"])
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers["authorization"]
        seen.append(auth)
        if auth == "Bearer key-a":
            return httpx.Response(429, text="tokens per day exceeded")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "model": "openai/gpt-oss-120b",
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = GroqGateway(
        "ignored",
        "openai/gpt-oss-120b",
        max_retries=1,
        http=http,
        key_pool=pool,
    )
    response = await gateway.complete([{"role": "user", "content": "hi"}])
    assert response.content == "ok"
    assert seen == ["Bearer key-a", "Bearer key-b"]
    assert pool.current() == "key-b"
    assert pool.current() == "key-b"


@pytest.mark.asyncio
async def test_groq_gateway_raises_when_all_keys_exhausted(monkeypatch):
    from app.infrastructure.llm import keys as keys_module

    now = [1_000_000.0]
    monkeypatch.setattr(keys_module, "_now_utc", lambda: now[0])
    monkeypatch.setattr(keys_module, "_next_midnight", lambda: now[0] + 1)
    pool = GroqKeyPool(["key-a", "key-b"])
    pool.mark_exhausted("key-a")
    pool.mark_exhausted("key-b")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="tokens per day (TPD) limit exceeded")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = GroqGateway(
        "ignored",
        "openai/gpt-oss-120b",
        max_retries=2,
        http=http,
        key_pool=pool,
    )
    with pytest.raises(LLMGatewayError) as exc_info:
        await gateway.complete([{"role": "user", "content": "hi"}])
    assert "exhausted" in str(exc_info.value)


@pytest.mark.asyncio
async def test_groq_gateway_retries_same_key_on_per_minute_429():
    """A 429 that names the per-minute (TPM) window must NOT park the key —
    it retries on the same key once the window rolls over."""
    from app.infrastructure.llm import keys as keys_module
    from app.infrastructure.llm.keys import GroqKeyPool

    now = [1_000_000.0]
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(keys_module, "_now_utc", lambda: now[0])
    monkeypatch.setattr(keys_module, "_next_midnight", lambda: now[0] + 1)

    pool = GroqKeyPool(["key-a", "key-b"])
    seen: list[str] = []
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429, text="Request too large for model `x` ... tokens per minute (TPM): Limit 8000, Requested 8200"
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = GroqGateway(
        "ignored",
        "openai/gpt-oss-120b",
        max_retries=1,
        http=http,
        key_pool=pool,
        skip_seconds=600,
    )
    gateway._tpm_retry_seconds = 0.1
    response = await gateway.complete([{"role": "user", "content": "hi"}])
    assert response.content == "ok"
    assert seen == ["Bearer key-a", "Bearer key-a"]
    assert pool.current() == "key-a"


@pytest.mark.asyncio
async def test_failover_uses_backup_when_primary_fails():
    primary = _FakeGateway(error=LLMGatewayError("primary exploded"))
    backup = _FakeGateway(content="from backup")
    gateway = FailoverGateway(primary, backup)
    response = await gateway.complete([{"role": "user", "content": "hi"}])
    assert response.content == "from backup"
    assert primary.calls == 1
    assert backup.calls == 1


@pytest.mark.asyncio
async def test_failover_does_not_touch_backup_on_success():
    primary = _FakeGateway(content="primary ok")
    backup = _FakeGateway(content="from backup")
    gateway = FailoverGateway(primary, backup)
    response = await gateway.complete([{"role": "user", "content": "hi"}])
    assert response.content == "primary ok"
    assert primary.calls == 1
    assert backup.calls == 0


@pytest.mark.asyncio
async def test_failover_raises_when_both_fail():
    primary = _FakeGateway(error=LLMGatewayError("primary exploded"))
    backup = _FakeGateway(error=LLMGatewayError("backup exploded"))
    gateway = FailoverGateway(primary, backup)
    with pytest.raises(LLMGatewayError) as exc_info:
        await gateway.complete([{"role": "user", "content": "hi"}])
    assert "backup exploded" in str(exc_info.value)
    assert "primary exploded" in str(exc_info.value)
    assert primary.calls == 1
    assert backup.calls == 1


@pytest.mark.asyncio
async def test_413_tpm_from_groq_retried_with_long_pause():
    requested_models: list[str] = []
    groq_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested_models.append(body["model"])
        if body["model"] == "model-x":
            groq_calls["n"] += 1
            if groq_calls["n"] == 1:
                return httpx.Response(413, text="Request too large for model `x` in organization `org` service tier `on_demand` on tokens per minute (TPM): Limit 1000, current 1500")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = GroqGateway(
        "model-key",
        "model-x",
        max_retries=1,
        http=http,
        skip_seconds=600,
    )
    gateway._tpm_retry_seconds = 0.1
    first = await gateway.complete([{"role": "user", "content": "hi"}])
    assert first.content == "ok"
    assert requested_models == ["model-x", "model-x"]


@pytest.mark.asyncio
async def test_413_without_tpm_is_hard_error():
    requested_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested_models.append(body["model"])
        return httpx.Response(413, text="context length exceeded")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = OpenRouterGateway(
        "test-key",
        "model-x",
        max_retries=1,
        fallback_models=["fallback-a"],
        http=http,
        skip_seconds=600,
    )
    with pytest.raises(LLMGatewayError) as exc_info:
        await gateway.complete([{"role": "user", "content": "hi"}])
    assert "413" in str(exc_info.value)
    assert requested_models == ["model-x", "fallback-a"]


@pytest.mark.asyncio
async def test_rate_limit_429_gets_short_skip_not_full_window():
    requested_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested_models.append(body["model"])
        if body["model"] == "model-x":
            return httpx.Response(429, text="rate limited")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = OpenRouterGateway(
        "test-key",
        "model-x",
        max_retries=1,
        fallback_models=["fallback-a"],
        http=http,
        skip_seconds=600,
        rate_limit_skip_seconds=60,
    )
    first = await gateway.complete([{"role": "user", "content": "hi"}])
    assert first.content == "ok"
    assert requested_models == ["model-x", "model-x", "fallback-a"]
    penalty = gateway._skip_until["model-x"] - time.monotonic()
    assert 0 < penalty <= 61
    await gateway.complete([{"role": "user", "content": "hi"}])
    assert requested_models == ["model-x", "model-x", "fallback-a", "fallback-a"]


@pytest.mark.asyncio
async def test_404_uses_full_skip_window():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model not found")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = OpenRouterGateway(
        "test-key",
        "model-x",
        max_retries=1,
        fallback_models=["fallback-a"],
        http=http,
        skip_seconds=600,
        rate_limit_skip_seconds=60,
    )
    with pytest.raises(LLMGatewayError):
        await gateway.complete([{"role": "user", "content": "hi"}])
    penalty = gateway._skip_until["model-x"] - time.monotonic()
    assert penalty > 590
