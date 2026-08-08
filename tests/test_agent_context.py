"""Agent context manager tests: profile/watchlist/memories/history assembly."""

import pytest

from app.application.agent.context import SYSTEM_PROMPT, build_messages, build_system_prompt


def test_system_prompt_has_integration_honesty_guardrail():
    """The agent must never claim email/calendar/Drive actions it cannot do."""
    assert "Gmail" in SYSTEM_PROMPT
    assert "Google Calendar" in SYSTEM_PROMPT
    assert "may or may not be connected" in SYSTEM_PROMPT
    assert "connect_google" in SYSTEM_PROMPT
    assert "Never claim to have sent email" in SYSTEM_PROMPT
    assert build_system_prompt() == SYSTEM_PROMPT


def test_system_prompt_mentions_connector_tools_when_connected():
    assert "read_google_sheet" in SYSTEM_PROMPT
    assert "Public Google Sheets" in SYSTEM_PROMPT
    for tool in ("search_emails", "find_calendar_events", "schedule_meeting", "read_drive_doc"):
        assert tool in SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_context_injects_connected_accounts(uow, demo_user):
    from app.domain.enums import IntegrationProvider

    user_id = demo_user["user_id"]
    async with uow:
        await uow.integrations.upsert(
            user_id, provider=IntegrationProvider.GMAIL, access_token="tok", scopes=["read"]
        )
        await uow.integrations.upsert(
            user_id, provider=IntegrationProvider.CALENDAR, access_token="tok", scopes=["read"]
        )
        conversation = await uow.conversations.create(user_id)
        await uow.commit()
        conversation_id = conversation.id

    async with uow:
        messages = await build_messages(uow, user_id=user_id, conversation_id=conversation_id)
    content = " ".join(m["content"] for m in messages)
    assert "User connected accounts: gmail, calendar" in content


@pytest.mark.asyncio
async def test_context_omits_connected_accounts_when_none(uow, demo_user):
    user_id = demo_user["user_id"]
    async with uow:
        conversation = await uow.conversations.create(user_id)
        await uow.commit()
        conversation_id = conversation.id

    async with uow:
        messages = await build_messages(uow, user_id=user_id, conversation_id=conversation_id)
    content = " ".join(m["content"] for m in messages)
    assert "User connected accounts:" not in content


@pytest.mark.asyncio
async def test_context_includes_user_signal(uow, demo_user):
    user_id = demo_user["user_id"]
    async with uow:
        conversation = await uow.conversations.create(user_id)
        await uow.conversations.add_message(
            conversation.id, role="user", content="what is aapl doing?", content_type="text"
        )
        await uow.commit()
        conversation_id = conversation.id

    async with uow:
        messages = await build_messages(uow, user_id=user_id, conversation_id=conversation_id)
    content = " ".join(m["content"] for m in messages)
    assert messages[0]["role"] == "system"
    # profile role "Investor" from demo_user fixture
    assert "Investor" in content
    assert "what is aapl doing?" in content
    assert "User watchlist:" not in content  # empty watchlist -> omitted


@pytest.mark.asyncio
async def test_context_includes_watchlist_and_memories(uow, session_factory, demo_user):
    user_id = demo_user["user_id"]
    async with uow:
        await uow.watchlist.add(user_id, symbol="AAPL", name="Apple", sector="Tech")
        await uow.memories.upsert_observation(
            user_id,
            memory_key="interest:ai",
            value={},
            summary="Interested in AI and semiconductors",
            confidence=0.9,
        )
        conversation = await uow.conversations.create(user_id)
        await uow.commit()
        conversation_id = conversation.id

    async with uow:
        messages = await build_messages(uow, user_id=user_id, conversation_id=conversation_id)
        content = " ".join(m["content"] for m in messages)
    assert "AAPL" in content
    assert "Interested in AI and semiconductors" in content


@pytest.mark.asyncio
async def test_context_keeps_full_media_content(uow, demo_user):
    """Regression: the 400-char history cap must not truncate media messages —
    the vision/STT extraction is the message's whole point, and its first chars
    may be a model's reasoning block rather than the content."""
    user_id = demo_user["user_id"]
    async with uow:
        conversation = await uow.conversations.create(user_id)
        long = "[image contents]\n<think>reasoning that must not survive</think>\n" + (
            "chart detail: sales by year and country " * 40
        )
        assert len(long) > 400
        await uow.conversations.add_message(
            conversation.id, role="user", content=long, content_type="image"
        )
        await uow.commit()
        conversation_id = conversation.id

    async with uow:
        messages = await build_messages(uow, user_id=user_id, conversation_id=conversation_id)
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    assert user_msgs and user_msgs[0] == long


@pytest.mark.asyncio
async def test_context_skips_empty_and_tool_messages(uow, demo_user):
    user_id = demo_user["user_id"]
    async with uow:
        conversation = await uow.conversations.create(user_id)
        await uow.conversations.add_message(
            conversation.id, role="assistant", content="", content_type="text"
        )
        await uow.conversations.add_message(
            conversation.id, role="assistant", content="  ", content_type="text"
        )
        await uow.conversations.add_message(
            conversation.id, role="tool", content="tool out", content_type="text"
        )
        await uow.conversations.add_message(
            conversation.id, role="assistant", content="a real reply", content_type="text"
        )
        await uow.commit()
        conversation_id = conversation.id

    async with uow:
        messages = await build_messages(
            uow,
            user_id=user_id,
            conversation_id=conversation_id,
        )
    contents = [m["content"] for m in messages]
    assert "a real reply" in contents
    assert "tool out" not in contents
    assert contents.count("a real reply") == 1


@pytest.mark.asyncio
async def test_context_media_placeholder_for_voice(uow, demo_user):
    user_id = demo_user["user_id"]
    async with uow:
        conversation = await uow.conversations.create(user_id)
        await uow.conversations.add_message(
            conversation.id,
            role="user",
            content=None,
            content_type="voice",
            media_meta={"file_id": "f1", "mime_type": "audio/ogg"},
        )
        await uow.commit()
        conversation_id = conversation.id

    async with uow:
        messages = await build_messages(uow, user_id=user_id, conversation_id=conversation_id)
    contents = [m["content"] for m in messages]
    assert any("voice message" in c for c in contents), contents


@pytest.mark.asyncio
async def test_context_truncates_long_history_messages(uow, demo_user):
    """Long replies must be trimmed so cheap model context budgets are not
    blown (OpenRouter free routes reject prompts above their input cap)."""
    user_id = demo_user["user_id"]
    async with uow:
        conversation = await uow.conversations.create(user_id)
        await uow.conversations.add_message(
            conversation.id,
            role="assistant",
            content="x" * 5000,
            content_type="text",
        )
        await uow.commit()
        conversation_id = conversation.id

    async with uow:
        messages = await build_messages(uow, user_id=user_id, conversation_id=conversation_id)
    history = [m for m in messages if m["role"] == "assistant"]
    assert history, "history message must be present"
    assert len(history[0]["content"]) <= 410, "long content must be truncated"
    assert history[0]["content"].endswith("…")


@pytest.mark.asyncio
async def test_context_respects_message_window(uow, demo_user):
    user_id = demo_user["user_id"]
    async with uow:
        conversation = await uow.conversations.create(user_id)
        for i in range(12):
            await uow.conversations.add_message(
                conversation.id, role="user", content=f"msg {i}", content_type="text"
            )
        await uow.commit()
        conversation_id = conversation.id

    async with uow:
        messages = await build_messages(
            uow, user_id=user_id, conversation_id=conversation_id, max_messages=6
        )
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    assert user_msgs == [f"msg {i}" for i in range(6, 12)], "only the latest 6 messages"


def test_system_prompt_is_stable():
    prompt = build_system_prompt()
    assert "Atlas" in prompt
    assert "tools" in prompt.lower()
    assert prompt.strip()
