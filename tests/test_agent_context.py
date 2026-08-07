"""Agent context manager tests: profile/watchlist/memories/history assembly."""

import pytest

from app.application.agent.context import SYSTEM_PROMPT, build_messages, build_system_prompt


def test_system_prompt_has_integration_honesty_guardrail():
    """The agent must never claim email/calendar/Drive actions it cannot do."""
    assert "Gmail" in SYSTEM_PROMPT
    assert "Google Calendar" in SYSTEM_PROMPT
    assert "not connected yet" in SYSTEM_PROMPT
    assert "Never claim to have sent email" in SYSTEM_PROMPT
    assert build_system_prompt() == SYSTEM_PROMPT


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


def test_system_prompt_is_stable():
    prompt = build_system_prompt()
    assert "Atlas" in prompt
    assert "tools" in prompt.lower()
    assert prompt.strip()
