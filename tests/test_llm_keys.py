"""GroqKeyPool: one key per request, switch only when the active key is
exhausted (429), and auto-recovery after the daily reset."""

import time

import pytest

from app.infrastructure.llm import keys as keys_module
from app.infrastructure.llm.keys import GroqKeyPool


def test_pool_returns_primary_first():
    pool = GroqKeyPool(["k1", "k2", "k3"])
    assert pool.current() == "k1"
    assert pool.current() == "k1"


def test_pool_switches_only_after_exhaustion():
    pool = GroqKeyPool(["k1", "k2"])
    pool.mark_exhausted("k1")
    assert pool.current() == "k2"
    assert pool.current() == "k2"


def test_pool_all_exhausted_returns_none(monkeypatch):
    now = [1_000_000.0]
    monkeypatch.setattr(keys_module, "_now_utc", lambda: now[0])
    monkeypatch.setattr(keys_module, "_next_midnight", lambda: now[0] + 1)
    pool = GroqKeyPool(["k1", "k2"])
    pool.mark_exhausted("k1")
    pool.mark_exhausted("k2")
    assert pool.current() is None


def test_pool_recovers_primary_after_daily_reset(monkeypatch):
    now = [1_000_000.0]
    monkeypatch.setattr(keys_module, "_now_utc", lambda: now[0])
    monkeypatch.setattr(keys_module, "_next_midnight", lambda: now[0] + 86_400)
    pool = GroqKeyPool(["k1", "k2"])
    pool.mark_exhausted("k1")
    assert pool.current() == "k2"
    now[0] += 86_400 + 1
    assert pool.current() == "k1"


def test_pool_deduplicates_and_requires_at_least_one_key():
    pool = GroqKeyPool(["k1", "k1", "", "k2"])
    assert pool.keys == ["k1", "k2"]
    with pytest.raises(ValueError):
        GroqKeyPool([])
    with pytest.raises(ValueError):
        GroqKeyPool(None)


def test_pool_marks_missing_key_silently():
    pool = GroqKeyPool(["k1", "k2"])
    pool.mark_exhausted("nope")
    assert pool.current() == "k1"
    assert time.monotonic() > 0  # keep the import useful


def test_is_groq_daily_cap_429_parks_only_daily_bucket():
    from app.infrastructure.llm.keys import is_groq_daily_cap_429

    assert is_groq_daily_cap_429(429, "tokens per day (TPD) limit 200000/200000") is True
    assert is_groq_daily_cap_429(429, "requests per day (RPD) limit reached") is True
    assert (
        is_groq_daily_cap_429(
            429, "tokens per minute (TPM): Limit 8000, Requested 13119"
        )
        is False
    )
    assert is_groq_daily_cap_429(429, "rate limited") is False
    assert is_groq_daily_cap_429(429, "") is False
    assert is_groq_daily_cap_429(200, "tokens per day") is False
    assert is_groq_daily_cap_429(None, "") is False
