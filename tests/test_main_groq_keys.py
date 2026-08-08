from app.core.config import Settings
from app.main import _groq_keys


def test_groq_keys_from_groq_api_keys_parsed_list():
    s = Settings(_env_file=None, groq_api_keys=["gsk_a", "gsk_b"])
    assert _groq_keys(s) == ["gsk_a", "gsk_b"]


def test_groq_keys_falls_back_to_single_key():
    s = Settings(_env_file=None, groq_api_keys=[], groq_api_key="gsk_zzz")
    assert _groq_keys(s) == ["gsk_zzz"]


def test_groq_keys_splits_comma_joined_single_key():
    s = Settings(_env_file=None, groq_api_keys=[], groq_api_key="gsk_a,gsk_b,gsk_c")
    assert _groq_keys(s) == ["gsk_a", "gsk_b", "gsk_c"]


def test_groq_keys_trims_whitespace_and_quotes():
    s = Settings(
        _env_file=None,
        groq_api_keys=["gsk_a, gsk_b", '"gsk_c"', " 'gsk_d' "],
    )
    assert _groq_keys(s) == ["gsk_a", "gsk_b", "gsk_c", "gsk_d"]


def test_groq_keys_default_fallback_when_nothing_set():
    s = Settings(_env_file=None, groq_api_keys=[], groq_api_key="")
    assert _groq_keys(s) == [
        "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"
    ]


def test_build_chat_gateway_chain_groq_gemini_openrouter():
    """With a Gemini key configured the chain becomes Groq -> Gemini ->
    OpenRouter, so Gemini absorbs the turn before the OpenRouter free route."""
    from app.infrastructure.llm.gateway import FailoverGateway, GeminiGateway
    from app.main import _build_chat_gateway

    s = Settings(
        _env_file=None,
        groq_api_keys=["gsk_a"],
        openrouter_api_key="or_test",
        gemini_api_key="AIza_gem",
    )
    gateway = _build_chat_gateway(s)
    assert isinstance(gateway, FailoverGateway)
    backup = gateway._backup
    assert isinstance(backup, FailoverGateway)
    assert isinstance(backup._primary, GeminiGateway)
    assert backup._primary._base_url == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )


def test_build_chat_gateway_skips_gemini_without_key():
    from app.infrastructure.llm.gateway import FailoverGateway
    from app.main import _build_chat_gateway

    s = Settings(_env_file=None, groq_api_keys=["gsk_a"], openrouter_api_key="or_test")
    gateway = _build_chat_gateway(s)
    assert isinstance(gateway, FailoverGateway)
    backup = gateway._backup
    assert isinstance(backup, FailoverGateway) is False
