"""OpenRouter gateway tests: payload shape, tool-call parsing, retries, errors."""

import json

import httpx
import pytest

from app.infrastructure.llm.gateway import (
    LLMGatewayError,
    LLMGatewayTransientError,
    OpenRouterGateway,
)


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
