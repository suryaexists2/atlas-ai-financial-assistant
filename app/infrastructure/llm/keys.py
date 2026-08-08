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


def is_groq_daily_cap_429(status_code: int | None, body: str) -> bool:
    """True when a Groq HTTP 429 means the *daily* bucket (TPD/RPD) is done.

    Groq also returns 429 for per-minute TPM/RPM windows, which roll over in
    ~60s; parking the key for those would strand every key on a burst. Only
    bodies naming the daily bucket park the key; everything else is treated as
    a transient window that retries on the same key.
    """
    if status_code != 429:
        return False
    low = (body or "").lower()
    return any(
        marker in low
        for marker in ("tokens per day", "requests per day", "(tpd)", "(rpd)")
    )


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
        """Park ``key`` (daily rate-limited) until the next UTC midnight."""
        self.park(key, _next_midnight())

    def park(self, key: str, until_epoch: float) -> None:
        """Park ``key`` until ``until_epoch``.

        Daily caps use the next UTC midnight; per-minute TPM/RPM windows use
        ~60s from now, so the burst rolls over to another key without the
        primary being stranded for the rest of the day.
        """
        for idx, candidate in enumerate(self._keys):
            if candidate == key:
                self._exhausted_until[idx] = until_epoch
                logger.warning(
                    "groq_key_parked",
                    key_index=idx,
                    total_keys=len(self._keys),
                    until=until_epoch,
                )
                return

    def park_for(self, key: str, seconds: float) -> None:
        """Park ``key`` for ``seconds`` from the pool clock (UTC epoch)."""
        self.park(key, _now_utc() + seconds)
