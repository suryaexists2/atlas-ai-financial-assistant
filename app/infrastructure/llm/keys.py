"""Groq API-key pool with lazy failover.

The Groq free tier caps tokens per day *per account* (the organization that
owns the key), so a handful of keys each give an independent daily quota.
Atlas keeps using one key and only switches to the next when the current key's
limits are exhausted (HTTP 429) — keys are never rotated per request.

Exhausted keys are parked until the next UTC midnight (when Groq resets daily
token caps), so the primary key resumes automatically the next day without a
restart.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

from app.core.logging import get_logger

logger = get_logger(__name__)


def _now_utc() -> float:
    return time.time()


def _next_midnight() -> float:
    now = datetime.now(UTC)
    tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return tomorrow.timestamp()


class GroqKeyPool:
    """Sequential failover over Groq API keys.

    Keys are used in the order they were given: the first (primary) key stays
    active for every request until `mark_exhausted` parks it (on a rate
    limit); the next key then takes over. `current()` restores the earliest
    key as soon as its cooldown has passed, so the primary key comes back
    automatically after the daily limit reset — no per-request rotation.
    """

    def __init__(self, keys: list[str]) -> None:
        deduped: list[str] = []
        for key in keys or []:
            if key and key not in deduped:
                deduped.append(key)
        if not deduped:
            raise ValueError("GroqKeyPool needs at least one key")
        self._keys = deduped
        # index -> epoch until which the key is parked (exhausted).
        self._exhausted_until: dict[int, float] = {}

    @property
    def keys(self) -> list[str]:
        return list(self._keys)

    def current(self) -> str | None:
        """The highest-priority usable key, or ``None`` when every key is
        exhausted until the next daily reset."""
        now = _now_utc()
        for idx, key in enumerate(self._keys):
            if self._exhausted_until.get(idx, 0.0) <= now:
                return key
        return None

    def mark_exhausted(self, key: str) -> None:
        """Park ``key`` (rate-limited) until the next UTC midnight."""
        for idx, candidate in enumerate(self._keys):
            if candidate == key:
                self._exhausted_until[idx] = _next_midnight()
                logger.warning(
                    "groq_key_exhausted",
                    key_index=idx,
                    total_keys=len(self._keys),
                    reset_at=self._exhausted_until[idx],
                )
                return
