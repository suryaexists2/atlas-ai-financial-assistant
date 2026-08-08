import pytest

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


def test_groq_keys_empty_when_nothing_set():
    s = Settings(_env_file=None, groq_api_keys=[], groq_api_key="")
    assert _groq_keys(s) == []