"""Deterministic intent router tests."""

import pytest

from app.application.router import (
    ALERTS,
    COMPLEX,
    CONFIRM,
    DOCUMENTS,
    GOOGLE,
    GREETING,
    IDENTITY,
    MARKET,
    REMINDERS,
    SCOPE,
    SCOPED_INTENTS,
    TEMPLATE_INTENTS,
    WATCHLIST,
    classify,
    template_reply,
)


@pytest.mark.parametrize(
    "text,intent",
    [
        ("hi", GREETING),
        ("Hello!", GREETING),
        ("good morning", GREETING),
        ("Hey Atlas", GREETING),
        ("thanks!", CONFIRM),
        ("thank you so much", CONFIRM),
        ("ok", CONFIRM),
        ("got it", CONFIRM),
        ("tell me a joke", SCOPE),
        ("write me a poem", SCOPE),
        ("what's the weather", SCOPE),
        ("translate this", SCOPE),
        ("give me a recipe", SCOPE),
        ("who are you?", IDENTITY),
        ("what can you do", IDENTITY),
        ("help", IDENTITY),
        ("are you a bot?", IDENTITY),
        ("what is apple stock price", MARKET),
        ("aapl quote?", MARKET),
        ("msft earnings", MARKET),
        ("nifty today", MARKET),
        ("show my watchlist", WATCHLIST),
        ("add nvda to watchlist", WATCHLIST),
        ("alert me when tsla drops", ALERTS),
        ("set price alert for msft", ALERTS),
        ("remind me at 10am", REMINDERS),
        ("schedule a meeting tomorrow", REMINDERS),
        ("create daily briefing", REMINDERS),
        ("read that pdf", DOCUMENTS),
        ("what's in that document?", DOCUMENTS),
        ("search my gmail for receipts", GOOGLE),
        ("show my calendar", GOOGLE),
        ("read my google sheet", GOOGLE),
        ("how should I structure my savings?", COMPLEX),
        ("how are you?", COMPLEX),
    ],
)
def test_classify(text, intent):
    assert classify(text) == intent


def test_classify_prefers_scope_over_market_keyword():
    assert classify("tell me a market joke") == SCOPE


def test_classify_prefers_fixed_over_google():
    assert classify("thanks for the calendar notes") == CONFIRM


def test_classify_media_without_text():
    assert classify("", is_media=True) == DOCUMENTS
    assert classify(None, is_media=True) == DOCUMENTS
    assert classify(None) == COMPLEX


def test_template_reply_coverage():
    for intent in TEMPLATE_INTENTS:
        assert template_reply(intent) is not None
    for intent in SCOPED_INTENTS | {COMPLEX}:
        assert template_reply(intent) is None


def test_all_scoped_intents_are_known_groups():
    # Every scoped intent must exist (tools.py maps these to schemas).
    assert "market" in SCOPED_INTENTS
    assert {"alerts", "watchlist", "reminders", "documents", "google"} <= SCOPED_INTENTS
