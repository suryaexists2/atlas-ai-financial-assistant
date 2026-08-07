"""Intelligence jobs: briefings, alerts, and reminders."""

import datetime as dt
import uuid

from app.application.intelligence import enqueue_for_user
from app.application.intelligence.alerts import (
    _matching_headline,
    _new_filing,
    _price_triggered,
    run_filing_alerts,
    run_news_alerts,
    run_price_alerts,
)
from app.application.intelligence.briefing import (
    _build_data_block,
    _deterministic_summary,
    _gather_interests,
    _match_interest_news,
)
from app.application.intelligence.jobs import ensure_cycle_jobs
from app.application.intelligence.reminders import fire_reminder
from app.domain.enums import AlertKind


def test_price_triggered_default_abs():
    assert _price_triggered({}, 6.0)
    assert _price_triggered(None, -6.5)
    assert not _price_triggered({}, 4.9)


def test_price_triggered_operators():
    assert _price_triggered({"operator": "gt", "percent": 5}, 6.0)
    assert not _price_triggered({"operator": "gt", "percent": 5}, 5.0)
    assert _price_triggered({"operator": "lt", "percent": 5}, -6.0)
    assert not _price_triggered({"operator": "lt", "percent": 5}, 4.0)


def test_matching_headline():
    items = [{"headline": "ACME beats estimates", "datetime": 1720000000}]
    assert _matching_headline(items, "ACME", since=None) == "ACME beats estimates"
    assert _matching_headline(items, "unrelated", since=None) is None
    future = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    assert _matching_headline(items, None, since=future) is None


def test_new_filing_returns_latest():
    items = [{"form": "8-K", "filed_on": "2026-08-06"}]
    assert _new_filing(items, since=None)["form"] == "8-K"
    old = dt.datetime(2026, 8, 10, tzinfo=dt.UTC)
    assert _new_filing(items, since=old) is None


def test_build_data_block_formats_quotes_and_news():
    quotes = [{"symbol": "AAPL", "name": "Apple", "c": 210.5, "dp": 1.25, "pc": 207.9}]
    news = [{"headline": "Fed holds rates", "source": "AP"}]
    block = _build_data_block(quotes, news)
    assert "AAPL" in block and "+1.25%" in block and "Fed holds rates" in block


def test_deterministic_summary_has_watchlist():
    assert "Morning brief" in _deterministic_summary([], [], [])
    item = type("Item", (), {"symbol": "MSFT", "name": "Microsoft"})
    quotes = [{"symbol": "MSFT", "c": 400.0, "dp": -0.5}]
    assert "MSFT" in _deterministic_summary([item()], quotes, [])


def test_deterministic_summary_interest_section():
    interest_news = [{"headline": "Semiconductors rally on AI demand"}]
    summary = _deterministic_summary([], [], [], ["ai", "semiconductors"], interest_news)
    assert "Your topics (ai, semiconductors)" in summary
    assert "Semiconductors rally on AI demand" in summary


def test_match_interest_news_filters_by_keywords():
    news = [
        {"headline": "AI model race heats up", "summary": ""},
        {"headline": "Bank earnings beat", "summary": ""},
        {"headline": "Chipmakers expand capacity", "summary": "semiconductor capex rises"},
    ]
    matched = _match_interest_news(news, ["AI", "semiconductor"])
    headlines = [n["headline"] for n in matched]
    assert headlines == ["AI model race heats up", "Chipmakers expand capacity"]


async def test_gather_interests_from_profile_and_memory(session_factory, uow):
    async with uow:
        user = await uow.users.create(telegram_id=2001, username="interest_user")
        await uow.profiles.upsert(user.id, interests=["AI", "technology"])
        await uow.memories.upsert_observation(
            user.id,
            memory_key="user_interests",
            value={"interests": ["macro", "AI"]},
            summary="interests",
            confidence=0.9,
        )
        await uow.commit()

    async with uow:
        topics = await _gather_interests(uow, user.id)
        assert topics == ["AI", "technology", "macro"]


async def test_price_alert_fires_once_then_cooldowns(session_factory, uow):
    async with uow:
        user = await uow.users.create(telegram_id=1001, username="price_user")
        await uow.alerts.create(
            user.id, kind=AlertKind.PRICE, symbol="AAPL", condition={"percent": 5}
        )
        await uow.commit()

    async with uow:

        class FakeFinnhub:
            async def quote(self, symbol):
                return {"price": 210.0, "d": 10.0, "dp": 5.5, "pc": 199.0}

        ctx = type("Ctx", (), {"finnhub": FakeFinnhub(), "sec": None, "gateway": None})()

        await run_price_alerts(uow, None, ctx)
        fired_first = await uow.outbox.claim_due(limit=10)
        assert len(fired_first) == 1
        assert fired_first[0].payload["text"].startswith("🔔 AAPL")
        for message in fired_first:
            await uow.outbox.mark_sent(message)

        await run_price_alerts(uow, None, ctx)
        fired_second = await uow.outbox.claim_due(limit=10)
        assert fired_second == []


async def test_news_alert_keyword_fires(session_factory, uow):
    async with uow:
        user = await uow.users.create(telegram_id=101, username="news_user")
        await uow.alerts.create(
            user.id, kind=AlertKind.NEWS, symbol="TSLA", condition={"keyword": "robotaxi"}
        )
        await uow.commit()

    async with uow:

        class FakeFinnhub:
            async def company_news(self, symbol, limit=5):
                return [{"headline": "Tesla unveils robotaxi fleet", "datetime": 1720000000}]

        ctx = type("Ctx", (), {"finnhub": FakeFinnhub(), "sec": None, "gateway": None})()
        await run_news_alerts(uow, None, ctx)
        fired = await uow.outbox.claim_due(limit=10)
        assert len(fired) == 1 and "robotaxi" in fired[0].payload["text"]


async def test_filing_alert_fires_on_sec(session_factory, uow):
    async with uow:
        user = await uow.users.create(telegram_id=102, username="filing_user")
        await uow.alerts.create(user.id, kind=AlertKind.FILING, symbol="NVDA")
        await uow.commit()

    async with uow:

        class FakeSec:
            async def recent_filings(self, symbol, form_types=None, limit=10):
                return [{"form": "10-Q", "filed_on": "2026-08-06"}]

        ctx = type("Ctx", (), {"finnhub": None, "sec": FakeSec(), "gateway": None})()
        assert await uow.outbox.claim_due(limit=10) == []

        await run_filing_alerts(uow, None, ctx)
        fired = await uow.outbox.claim_due(limit=10)
        assert len(fired) == 1 and "10-Q" in fired[0].payload["text"]


async def test_fire_reminder_enqueues_text(session_factory, uow):
    async with uow:
        user = await uow.users.create(telegram_id=103, username="rem_user")
        await uow.commit()

    async with uow:
        Job = type(
            "Job",
            (),
            {"user_id": user.id, "job_type": "reminder", "params": {"text": "quarterly review"}},
        )
        await fire_reminder(uow, Job(), None)
        fired = await uow.outbox.claim_due(limit=10)
        assert fired[0].payload["text"] == "⏰ Reminder: quarterly review"


async def test_enqueue_for_user_false_for_unknown_user(session_factory, uow):
    async with uow:
        Job = type("Job", (), {"user_id": uuid.uuid4(), "job_type": "x", "params": {}})
        assert not await enqueue_for_user(uow, Job(), "hello")


async def test_ensure_cycle_jobs_seeds_once(session_factory, uow):
    async with uow:
        await ensure_cycle_jobs(uow)
        first = {j.job_type for j in await uow.jobs.list_enabled()}
        assert {"price_alerts", "news_alerts", "filing_alerts"} <= first

        await ensure_cycle_jobs(uow)
        second = {j.job_type for j in await uow.jobs.list_enabled()}
        assert second == first
