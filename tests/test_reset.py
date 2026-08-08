"""/reset feature tests: two-step confirmation, full wipe, fresh onboarding.

The reset is deterministic and runs before onboarding/agent logic, so these
tests assert that no LLM turn happens and that every piece of user data is
hard-deleted while the users row survives.
"""

import datetime as dt

from sqlalchemy import func, select

from app.application.reset import is_confirmation, is_reset_command, reset_turn, wipe_user
from app.domain.entities import OAuthFlow
from app.domain.enums import AlertKind, IntegrationProvider, MessageRole, OnboardingStatus
from app.interfaces.telegram.normalized import NormalizedMessage
from app.interfaces.telegram.processor import ReplyContext
from app.interfaces.telegram.responder import AgentComposer

# ---- command / confirmation detection ----------------------------------------


def test_reset_command_detection():
    yes = ["/reset", "/RESET", "/reset ", "/reset!", "/reset .", "  /reset  "]
    no = ["/reset all", "reset", "hi /reset", "resetting", "", None, "/new"]
    for text in yes:
        assert is_reset_command(text), f"must match: {text!r}"
    for text in no:
        assert not is_reset_command(text), f"must NOT match: {text!r}"


def test_confirmation_detection():
    yes = ["yes", "Yes", "YES", "y", "yeah", "yep", "sure", "ok", "okay", "confirm", "haan", "ha"]
    no = ["no", "nope", "maybe", "yes please", "not yet", "cancel", "", None, "yes but later"]
    for text in yes:
        assert is_confirmation(text), f"must match: {text!r}"
    for text in no:
        assert not is_confirmation(text), f"must NOT match: {text!r}"


# ---- helpers ------------------------------------------------------------------


class FakeAgent:
    def __init__(self):
        self.calls = 0
        self.last_error = None

    async def run(self, uow, *, user_id, conversation_id, tool_context=None, intent="complex"):
        self.calls += 1
        self.last_intent = intent
        return "agent reply"


def make_ctx(text: str, user_id, conversation_id, uow) -> ReplyContext:
    message = NormalizedMessage(
        correlation_id="t-1",
        telegram_user_id=999999,
        chat_id=888888,
        update_id=1,
        message_id=1,
        source="webhook",
        text=text,
    )
    return ReplyContext(
        message=message,
        uow=uow,
        user_id=user_id,
        conversation_id=conversation_id,
    )


async def seed_user_data(uow, user_id):
    """Puts one row of every user-owned table on the account."""
    conv = await uow.conversations.create(user_id, title="old chat")
    await uow.conversations.add_message(conv.id, role=MessageRole.USER, content="hello")
    await uow.memories.upsert_observation(
        user_id, memory_key="role", value=None, summary="Investor", confidence=0.9
    )
    await uow.watchlist.add(user_id, symbol="TSLA", name="Tesla", sector=None)
    await uow.alerts.create(user_id, kind=AlertKind.PRICE, symbol="TSLA")
    await uow.jobs.create(job_type="daily_brief", cron_expr="0 8 * * *", user_id=user_id)
    await uow.documents.create(user_id, filename="report.pdf", mime_type="application/pdf")
    await uow.integrations.upsert(
        user_id, provider=IntegrationProvider.GMAIL, access_token="tok", scopes=["email"]
    )
    await uow.oauth_flows.create(
        state="s-1",
        user_id=user_id,
        chat_id=888888,
        code_verifier="v",
        expires_at=dt.datetime.now(dt.UTC) + dt.timedelta(hours=1),
    )


async def count_user_data(uow, user_id) -> dict:
    oauth_count = await uow.session.execute(
        select(func.count()).select_from(OAuthFlow).where(OAuthFlow.user_id == user_id)
    )
    return {
        "conversations": len(await uow.conversations.list_for_user(user_id, limit=100)),
        "memories": len(await uow.memories.list_active(user_id, limit=500)),
        "watchlist": len(await uow.watchlist.list_active(user_id)),
        "alerts": len(await uow.alerts.list_enabled(user_id)),
        "documents": len(await uow.documents.list_for_user(user_id)),
        "jobs": len([j for j in await uow.jobs.list_enabled() if j.user_id == user_id]),
        "google": (
            1 if await uow.integrations.get_by_provider(user_id, IntegrationProvider.GMAIL) else 0
        ),
        "oauth": oauth_count.scalar_one(),
    }


# ---- /reset state machine through the composer --------------------------------


async def test_reset_arms_confirmation_without_agent(uow, demo_user):
    async with uow:
        await uow.profiles.set_onboarding(
            demo_user["user_id"], OnboardingStatus.COMPLETED, {"step": "done"}
        )
        await uow.commit()
    agent = FakeAgent()
    composer = AgentComposer(agent)
    ctx = make_ctx("/reset", demo_user["user_id"], "cv-1", uow)
    async with uow:
        reply = await composer(ctx)
        assert reply is not None
        assert "Are you sure you want to reset" in reply
        assert agent.calls == 0, "no LLM turn for /reset"
    async with uow:
        profile = await uow.profiles.get_by_user_id(demo_user["user_id"])
        assert "reset_pending_at" in profile.onboarding_context
        # Pending flag must not disturb the onboarding step key.
        assert profile.onboarding_context["step"] == "done"


async def test_reset_reprompts_while_pending(uow, demo_user):
    async with uow:
        await uow.profiles.set_onboarding(
            demo_user["user_id"], OnboardingStatus.COMPLETED, {"step": "done"}
        )
        await uow.commit()
    agent = FakeAgent()
    composer = AgentComposer(agent)
    async with uow:
        await composer(make_ctx("/reset", demo_user["user_id"], "cv-1", uow))
        reply = await composer(make_ctx("/reset", demo_user["user_id"], "cv-1", uow))
        assert reply is not None
        assert "already have a reset pending" in reply
        assert agent.calls == 0


async def test_reset_confirmation_wipes_everything_and_starts_fresh(uow, demo_user):
    async with uow:
        await uow.profiles.set_onboarding(
            demo_user["user_id"], OnboardingStatus.COMPLETED, {"step": "done"}
        )
        await seed_user_data(uow, demo_user["user_id"])
        await uow.commit()

    agent = FakeAgent()
    composer = AgentComposer(agent)
    ctx = make_ctx("/reset", demo_user["user_id"], "cv-1", uow)
    async with uow:
        await composer(ctx)
    ctx = make_ctx("yes", demo_user["user_id"], "cv-1", uow)
    async with uow:
        reply = await composer(ctx)
        assert reply is not None
        assert "your data has been cleared" in reply
        assert agent.calls == 0
        assert ctx.assistant_persisted is True, "composer persisted into the fresh conversation"

    async with uow:
        counts = await count_user_data(uow, demo_user["user_id"])
        assert counts == {
            "conversations": 1,  # the fresh conversation created for the reply
            "memories": 0,
            "watchlist": 0,
            "alerts": 0,
            "documents": 0,
            "jobs": 0,
            "google": 0,
            "oauth": 0,
        }
        user = await uow.users.get_by_id(demo_user["user_id"])
        assert user is not None, "users row survives the wipe"
        profile = await uow.profiles.get_by_user_id(demo_user["user_id"])
        assert profile.onboarding_status == OnboardingStatus.NOT_STARTED
        assert profile.onboarding_context == {}
        assert profile.role is None
        assert profile.interests is None
        assert profile.briefing_time is None
        convos = await uow.conversations.list_for_user(demo_user["user_id"])
        assert len(convos) == 1
        messages = await uow.conversations.list_messages(convos[0].id)
        assert [m.role.value for m in messages] == ["assistant"]
        assert "your data has been cleared" in messages[0].content


async def test_reset_anything_else_cancels(uow, demo_user):
    async with uow:
        await uow.profiles.set_onboarding(
            demo_user["user_id"], OnboardingStatus.COMPLETED, {"step": "done"}
        )
        await seed_user_data(uow, demo_user["user_id"])
        await uow.commit()

    agent = FakeAgent()
    composer = AgentComposer(agent)
    async with uow:
        await composer(make_ctx("/reset", demo_user["user_id"], "cv-1", uow))
        reply = await composer(make_ctx("no", demo_user["user_id"], "cv-1", uow))
        assert reply is not None
        assert "Reset cancelled" in reply
        assert agent.calls == 0

    async with uow:
        counts = await count_user_data(uow, demo_user["user_id"])
        assert counts["conversations"] == 1
        assert counts["memories"] == 1
        assert counts["watchlist"] == 1
        assert counts["alerts"] == 1
        assert counts["documents"] == 1
        assert counts["jobs"] == 1
        assert counts["google"] == 1
        profile = await uow.profiles.get_by_user_id(demo_user["user_id"])
        assert "reset_pending_at" not in profile.onboarding_context
        assert profile.onboarding_status == OnboardingStatus.COMPLETED


async def test_expired_pending_confirmation_is_ignored(uow, demo_user):
    async with uow:
        await uow.profiles.set_onboarding(
            demo_user["user_id"], OnboardingStatus.COMPLETED, {"step": "done"}
        )
        await seed_user_data(uow, demo_user["user_id"])
        old = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=15)
        await uow.profiles.upsert(
            demo_user["user_id"],
            onboarding_context={"step": "done", "reset_pending_at": old.isoformat()},
        )
        await uow.commit()

    agent = FakeAgent()
    composer = AgentComposer(agent)
    async with uow:
        reply = await composer(make_ctx("yes", demo_user["user_id"], "cv-1", uow))
        # Stale "yes" must not wipe: it falls through to the agent like any message.
        assert reply == "agent reply"
        assert agent.calls == 1
    async with uow:
        counts = await count_user_data(uow, demo_user["user_id"])
        assert counts["memories"] == 1
        assert counts["google"] == 1
        profile = await uow.profiles.get_by_user_id(demo_user["user_id"])
        assert "reset_pending_at" not in profile.onboarding_context


async def test_normal_message_without_pending_still_runs_agent(uow, demo_user):
    async with uow:
        await uow.profiles.set_onboarding(
            demo_user["user_id"], OnboardingStatus.COMPLETED, {"step": "done"}
        )
        await uow.commit()
    agent = FakeAgent()
    composer = AgentComposer(agent)
    async with uow:
        reply = await composer(
            make_ctx("what is NVDA trading at?", demo_user["user_id"], "cv-1", uow)
        )
        assert reply == "agent reply"
        assert agent.calls == 1


async def test_reset_works_mid_onboarding(uow, demo_user):
    async with uow:
        await uow.profiles.set_onboarding(
            demo_user["user_id"], OnboardingStatus.IN_PROGRESS, {"step": "role"}
        )
        await uow.commit()
    agent = FakeAgent()
    composer = AgentComposer(agent)
    async with uow:
        reply = await composer(make_ctx("/reset", demo_user["user_id"], "cv-1", uow))
        assert reply is not None
        assert "Are you sure you want to reset" in reply


async def test_wipe_user_is_idempotent_for_fresh_account(uow, demo_user):
    async with uow:
        await wipe_user(uow, demo_user["user_id"])
        await uow.commit()
        profile = await uow.profiles.get_by_user_id(demo_user["user_id"])
        assert profile.onboarding_status == OnboardingStatus.NOT_STARTED


async def test_reset_turn_pending_flag_survives_other_profile_keys(uow, demo_user):
    async with uow:
        await uow.profiles.set_onboarding(
            demo_user["user_id"], OnboardingStatus.IN_PROGRESS, {"step": "watchlist"}
        )
        await uow.commit()
    async with uow:
        out = await reset_turn(uow, user_id=demo_user["user_id"], text="/reset")
        assert out.reply is not None
        assert out.wiped is False
    async with uow:
        profile = await uow.profiles.get_by_user_id(demo_user["user_id"])
        assert profile.onboarding_context["step"] == "watchlist"
        assert "reset_pending_at" in profile.onboarding_context
        assert profile.onboarding_status == OnboardingStatus.IN_PROGRESS

