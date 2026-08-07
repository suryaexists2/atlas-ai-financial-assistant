"""Free-model discovery registry tests: capability filter, refresh, config defaults."""

import httpx
import pytest

from app.core.config import Settings
from app.infrastructure.llm.models_registry import (
    FreeModelRegistry,
    filter_free_models,
)


def _models_payload() -> list[dict]:
    """Mix of compatible and incompatible models (shaped like OpenRouter v1)."""
    return [
        {
            "id": "vendor/model-a:free",
            "name": "Model A",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 262144,
            "top_provider": {"context_length": 262144, "max_completion_tokens": 32768},
            "supported_parameters": ["tools", "response_format", "reasoning"],
            "reasoning": {"mandatory": False, "default_enabled": False},
        },
        {
            "id": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "name": "Nemotron Ultra",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 1_000_000,
            "top_provider": {"context_length": 1000000, "max_completion_tokens": 65536},
            "supported_parameters": ["tools", "reasoning", "temperature"],
            "reasoning": {"mandatory": False, "default_enabled": True},
        },
        {
            "id": "vendor/paid-chat",
            "name": "Paid Chat",
            "pricing": {"prompt": "0.25", "completion": "1.0"},
            "context_length": 131072,
            "top_provider": {"max_completion_tokens": 32768},
            "supported_parameters": ["tools"],
        },
        {
            "id": "vendor/guard-1:free",
            "name": "Content Guardrail",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 128000,
            "supported_parameters": [],
        },
        {
            "id": "openai/gpt-oss-20b:free",
            "name": "GPT OSS 20B",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 131072,
            "top_provider": {"max_completion_tokens": 32768},
            "supported_parameters": ["tools", "reasoning"],
            "reasoning": {"mandatory": True},
        },
        {
            "id": "vendor/tiny:free",
            "name": "Tiny",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 4096,
            "top_provider": {"max_completion_tokens": 1024},
            "supported_parameters": ["tools"],
        },
        {
            "id": "vendor/small-cap:free",
            "name": "Small Cap",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 128000,
            "top_provider": {"max_completion_tokens": 256},
            "supported_parameters": ["tools"],
        },
        {
            "id": "vendor/text-embed-3:free",
            "name": "Embedder",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 128000,
            "top_provider": {"max_completion_tokens": 32768},
            "supported_parameters": ["tools"],
        },
        {
            "id": "vendor/north-mini-code:free",
            "name": "Code Agent",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 256000,
            "top_provider": {"max_completion_tokens": 64000},
            "supported_parameters": ["tools"],
        },
    ]


def test_filter_keeps_only_compatible_free_models():
    result = filter_free_models(_models_payload())
    assert [m.id for m in result] == [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "vendor/model-a:free",
    ]


def test_filter_requires_tools_capability():
    payload = [
        {
            "id": "vendor/no-tools:free",
            "name": "No Tools",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 262144,
            "top_provider": {"max_completion_tokens": 32768},
            "supported_parameters": ["temperature"],
        }
    ]
    assert filter_free_models(payload) == []


def test_filter_excludes_mandatory_reasoning():
    payload = [
        {
            "id": "vendor/must-reason:free",
            "name": "Must Reason",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 262144,
            "top_provider": {"max_completion_tokens": 32768},
            "supported_parameters": ["tools"],
            "reasoning": {"mandatory": True},
        }
    ]
    assert filter_free_models(payload) == []


def test_filter_excludes_paid_and_tiny_context_models():
    payload = [
        {
            "id": "vendor/paid:free",
            "name": "Paid route",
            "pricing": {"prompt": "0.1", "completion": "0.2"},
            "context_length": 262144,
            "supported_parameters": ["tools"],
        },
        {
            "id": "vendor/small:free",
            "name": "Small",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 4096,
            "supported_parameters": ["tools"],
        },
    ]
    assert filter_free_models(payload) == []


def test_filter_marks_reasoning_disablable():
    result = {m.id: m for m in filter_free_models(_models_payload())}
    assert result["vendor/model-a:free"].reasoning_disablable is True
    assert result["nvidia/nemotron-3-ultra-550b-a55b:free"].reasoning_disablable is True


def _registry_with(handler, **kwargs):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return FreeModelRegistry(http=http, **kwargs)


@pytest.mark.asyncio
async def test_registry_refresh_populates_extras():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"data": _models_payload()})

    registry = _registry_with(handler)
    assert registry.extra_models() == []
    assert await registry.refresh() is True
    assert calls["n"] == 1
    assert [m.id for m in registry.extra_models()] == [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "vendor/model-a:free",
    ]


@pytest.mark.asyncio
async def test_registry_failed_refresh_keeps_last_known_list():
    state = {"fail": True}

    def handler(request):
        if state["fail"]:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"data": _models_payload()})

    registry = _registry_with(handler)
    assert await registry.refresh() is False
    assert registry.extra_models() == []
    registry._last_refresh = None
    state["fail"] = False
    assert await registry.refresh() is True
    assert len(registry.extra_models()) == 2
    registry._last_refresh = None
    state["fail"] = True
    assert await registry.refresh() is False
    assert len(registry.extra_models()) == 2


@pytest.mark.asyncio
async def test_registry_ensure_fresh_respects_interval():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"data": _models_payload()})

    registry = _registry_with(handler, refresh_seconds=3600)
    await registry.ensure_fresh()
    assert calls["n"] == 1
    await registry.ensure_fresh()
    assert calls["n"] == 1
    registry._last_refresh = None
    await registry.ensure_fresh()
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_registry_mock_models_match_live_snapshot_fields():
    # Guard against drifting expectations: the filter must still pass a model
    # shaped exactly like today's live catalogue entries.
    payload = [
        {
            "id": "google/gemma-4-31b-it:free",
            "name": "Gemma 4 31B",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 262144,
            "top_provider": {"context_length": 262144, "max_completion_tokens": 32768},
            "supported_parameters": [
                "include_reasoning",
                "max_tokens",
                "reasoning",
                "response_format",
                "seed",
                "temperature",
                "tool_choice",
                "tools",
                "top_p",
            ],
            "reasoning": {"mandatory": False, "default_enabled": False},
        }
    ]
    result = filter_free_models(payload)
    assert len(result) == 1
    assert result[0].id == "google/gemma-4-31b-it:free"
    assert result[0].reasoning_disablable is True


def test_config_defaults_resilient_chain():
    settings = Settings(_env_file=None)
    chain = [settings.llm_model, *settings.llm_fallback_models]
    free = [m for m in chain if m.endswith(":free")]
    assert len(free) >= 3
    assert "google/gemini-2.0-flash" not in chain
    assert "google/gemini-2.5-flash" not in chain
    assert "meta-llama/llama-3.1-8b-instruct" not in chain
    assert settings.llm_max_tokens == 600
    assert settings.llm_dynamic_free_models is True
    assert settings.llm_model_skip_seconds == 600
    assert settings.llm_free_min_context == 32_000


def test_config_defaults_groq_stack():
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "groq"
    assert settings.groq_llm_model == "llama-3.3-70b-versatile"
    assert settings.groq_llm_fallback is None
    assert settings.stt_provider == "groq"
    assert settings.groq_stt_model == "whisper-large-v3-turbo"
    assert settings.vision_model == "google/gemma-4-26b-a4b-it:free"
