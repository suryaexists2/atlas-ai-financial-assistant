"""Conversational onboarding tests: flow, skip behavior, parsing, persistence."""

from app.application.onboarding import (
    OnboardingEngine,
    _parse_interests,
    _parse_role,
    _parse_symbols,
    _parse_time,
    _wants_agent,
)
from app.domain.enums import OnboardingStatus


async def make_user(uow):
    user = await uow.users.create(telegram_id=555777, username="ada", timezone="UTC")
    await uow.profiles.upsert(user.id)
    await uow.commit()
    return user.id


async def run_flow(uow, user_id, replies, engine=None):
    """Walks onboarding through a list of user replies, returning turns."""
    engine = engine or OnboardingEngine()
    out = []
    for text in replies:
        out.append(await engine.turn(uow, user_id=user_id, text=text))
    return out


async def test_onboarding_runs_to_completion_and_persists(uow):
    async with uow:
        user_id = await make_user(uow)
        turns = await run_flow(
            uow,
            user_id,
            ["I'm an Investor", "semiconductors, earnings", "AAPL NVDA", "8:30am", "skip"],
        )
        assert len(turns) >= 4
        assert turns[0].text is not None  # welcome question
        assert turns[-1].text is not None  # done message
        assert turns[-1].completed

        profile = await uow.profiles.get_by_user_id(user_id)
        assert profile.onboarding_status == OnboardingStatus.COMPLETED
        assert profile.role == "Investor"
        assert isinstance(profile.interests, list) and profile.interests
        assert profile.briefing_time == "08:30"

        watchlist = await uow.watchlist.list_active(user_id)
        symbols = {w.symbol for w in watchlist}
        assert {"AAPL", "NVDA"} <= symbols

        jobs = await uow.jobs.list_enabled()
        briefs = [j for j in jobs if j.user_id == user_id and j.job_type == "daily_brief"]
        assert len(briefs) == 1
        assert briefs[0].cron_expr == "30 8 * * *"

        memories = await uow.memories.list_active(user_id)
        keys = {m.memory_key for m in memories}
        assert "user_profile" in keys and "user_interests" in keys


async def test_onboarding_all_skips(uow):
    async with uow:
        user_id = await make_user(uow)
        engine = OnboardingEngine(default_briefing_time="09:00")
        turns = await run_flow(uow, user_id, ["skip"] * 6, engine)
        assert turns[-1].completed
        profile = await uow.profiles.get_by_user_id(user_id)
        assert profile.onboarding_status == OnboardingStatus.COMPLETED
        assert await uow.watchlist.list_active(user_id) == []
        assert profile.briefing_time == "09:00"


async def test_question_exits_onboarding_to_agent(uow):
    async with uow:
        user_id = await make_user(uow)
        engine = OnboardingEngine()
        turn = await engine.turn(uow, user_id=user_id, text="What can you do?")
        assert turn.completed
        assert turn.followed_by_agent
        assert turn.text is None
        profile = await uow.profiles.get_by_user_id(user_id)
        assert profile.onboarding_status == OnboardingStatus.COMPLETED


async def test_media_first_message_skips_onboarding(uow):
    async with uow:
        user_id = await make_user(uow)
        engine = OnboardingEngine()
        reply = await engine.turn(uow, user_id=user_id, text="", is_media=True)
        assert reply.completed
        assert reply.text
        profile = await uow.profiles.get_by_user_id(user_id)
        assert profile.onboarding_status == OnboardingStatus.COMPLETED


async def test_completed_profile_stays_completed(uow):
    async with uow:
        user_id = await make_user(uow)
        engine = OnboardingEngine()
        await uow.profiles.set_onboarding(user_id, OnboardingStatus.COMPLETED, {"step": "done"})
        await uow.commit()
        reply = await engine.turn(uow, user_id=user_id, text="what about Google?")
        assert reply.completed
        assert reply.text is None


async def test_google_connect_step_is_optional_and_skippable(uow):
    async with uow:
        user_id = await make_user(uow)
        engine = OnboardingEngine(google_connect_available=True)
        turns = await run_flow(
            uow,
            user_id,
            ["I'm an Analyst", "tech", "AAPL", "8:00am", "skip", "skip"],
            engine,
        )
        assert turns[-2].text is not None
        assert "connect your Google account" in turns[-2].text  # optional step shown
        assert turns[-2].still_onboarding
        assert turns[-1].completed
        assert "You're all set" in turns[-1].text
        profile = await uow.profiles.get_by_user_id(user_id)
        assert profile.onboarding_status == OnboardingStatus.COMPLETED


async def test_google_connect_step_accepts_yes(uow):
    async with uow:
        user_id = await make_user(uow)
        engine = OnboardingEngine(google_connect_available=True)
        turns = await run_flow(
            uow,
            user_id,
            ["Investor", "semiconductors", "NVDA", "9:00", "skip", "yes connect it"],
            engine,
        )
        assert turns[-1].completed
        assert "You're all set" in turns[-1].text


# --- parsing helpers ---------------------------------------------------------


def test_parse_role_extracts():
    assert _parse_role("I'm an Investor") == "Investor"
    assert _parse_role("i am a founder") == "Founder"


def test_parse_role_garbage_returns_none():
    assert _parse_role("gold is the only asset that matters") is None
    assert _parse_role("not now") is None


def test_parse_interests_extracts_known_and_free():
    assert "semiconductors" in _parse_interests("semiconductors, tech")
    assert _parse_interests("macros and economy") != []


def test_parse_symbols_uppercase_only():
    assert _parse_symbols("watch AAPL and NVDA") == ["AAPL", "NVDA"]
    assert _parse_symbols("watch aapl please") == []
    assert _parse_symbols("nothing important") == []


def test_parse_time():
    assert _parse_time("8:30am") == ("08:30", True)
    assert _parse_time("skip") == (None, False)
    assert _parse_time("random text") == (None, False)


def test_wants_agent():
    assert _wants_agent("What can you do?")
    assert not _wants_agent("I like tech")
