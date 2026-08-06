"""Centralized Telegram rate limiting (token bucket, per-chat + global)."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict


class TokenBucket:
    def __init__(self, *, rate_per_sec: float, burst: int) -> None:
        self._rate = rate_per_sec
        self._burst = burst
        self._tokens = float(burst)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Blocks until a token is available (respects the bucket's rate)."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated
                self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
                self._updated = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait = (1 - self._tokens) / self._rate
            await asyncio.sleep(max(0.0, wait))


class RateLimiter:
    """Two-level limiter: global bucket + one bucket per chat."""

    def __init__(self, *, global_per_sec: float, per_chat_per_sec: float, burst: int) -> None:
        self._global = TokenBucket(rate_per_sec=global_per_sec, burst=burst)
        self._per_chat: defaultdict[int, TokenBucket] = defaultdict(
            lambda: TokenBucket(rate_per_sec=per_chat_per_sec, burst=burst)
        )

    async def acquire(self, chat_id: int) -> None:
        await self._global.acquire()
        await self._per_chat[chat_id].acquire()
