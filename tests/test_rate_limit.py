"""Rate limiter tests: token bucket pacing + two-level limiter."""

import asyncio
import time

import pytest

from app.infrastructure.telegram.rate_limit import RateLimiter, TokenBucket


@pytest.mark.asyncio
async def test_token_bucket_immediate_acquires_within_burst():
    bucket = TokenBucket(rate_per_sec=10_000, burst=100)
    start = time.monotonic()
    for _ in range(10):
        await bucket.acquire()
    assert time.monotonic() - start < 1.0  # burst covers all 10 instantly


@pytest.mark.asyncio
async def test_token_bucket_refills_at_rate():
    bucket = TokenBucket(rate_per_sec=100, burst=1)
    await bucket.acquire()  # consumes the single token
    start = time.monotonic()
    await bucket.acquire()  # must wait ~0.01s for refill
    elapsed = time.monotonic() - start
    assert 0.005 <= elapsed <= 0.5


@pytest.mark.asyncio
async def test_rate_limiter_global_and_per_chat():
    limiter = RateLimiter(global_per_sec=10_000, per_chat_per_sec=10_000, burst=1)
    start = time.monotonic()
    await limiter.acquire(chat_id=111)
    await limiter.acquire(chat_id=222)  # different chat still gated by global burst
    assert time.monotonic() - start < 1.0


@pytest.mark.asyncio
async def test_rate_limiter_per_chat_blocks_second_message():
    limiter = RateLimiter(global_per_sec=100, per_chat_per_sec=100, burst=1)
    await limiter.acquire(chat_id=555)
    start = time.monotonic()
    await limiter.acquire(chat_id=555)  # same chat -> must wait for refill
    assert time.monotonic() - start >= 0.005


@pytest.mark.asyncio
async def test_concurrent_acquires_do_not_corrupt_bucket():
    limiter = RateLimiter(global_per_sec=1_000_000, per_chat_per_sec=1_000_000, burst=50)
    start = time.monotonic()
    await asyncio.gather(*[limiter.acquire(chat_id=777) for _ in range(50)])
    assert time.monotonic() - start < 1.0
