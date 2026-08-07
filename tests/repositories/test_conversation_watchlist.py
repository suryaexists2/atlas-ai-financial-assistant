"""Conversation, message and watchlist repository tests (M1)."""

import pytest

from app.domain.enums import ContentType, MessageRole


@pytest.mark.asyncio
async def test_conversation_and_message_roundtrip(uow, demo_user):
    user_id = demo_user["user_id"]

    async with uow:
        convo = await uow.conversations.create(user_id, title="Intro")
        await uow.commit()
        convo_id = convo.id

    async with uow:
        msg = await uow.conversations.add_message(
            convo_id,
            role=MessageRole.USER,
            content="Why did NVDA move today?",
        )
        await uow.commit()
        assert msg.content_type == ContentType.TEXT

    async with uow:
        msgs = await uow.conversations.list_messages(convo_id)
        assert len(msgs) == 1
        assert msgs[0].content == "Why did NVDA move today?"


@pytest.mark.asyncio
async def test_conversation_list_ordered_by_recency(uow, demo_user):
    user_id = demo_user["user_id"]
    ids = []
    async with uow:
        for i in range(3):
            convo = await uow.conversations.create(user_id, title=f"c{i}")
            ids.append(convo.id)
        await uow.commit()

    async with uow:
        convos = await uow.conversations.list_for_user(user_id, limit=10)
        assert [c.id for c in convos] == list(reversed(ids))


@pytest.mark.asyncio
async def test_messages_are_isolated_per_conversation(uow, demo_user):
    user_id = demo_user["user_id"]
    async with uow:
        c1 = await uow.conversations.create(user_id)
        c2 = await uow.conversations.create(user_id)
        await uow.conversations.add_message(c1.id, role=MessageRole.USER, content="only c1")
        await uow.conversations.add_message(c2.id, role=MessageRole.USER, content="only c2")
        await uow.commit()

    async with uow:
        assert len(await uow.conversations.list_messages(c1.id)) == 1
        assert len(await uow.conversations.list_messages(c2.id)) == 1


@pytest.mark.asyncio
async def test_list_messages_keeps_most_recent_window(uow, demo_user):
    """list_messages(limit=N) must return the LAST N messages in chronological
    order — the agent needs the newest user turn, not the oldest history."""
    user_id = demo_user["user_id"]
    async with uow:
        convo = await uow.conversations.create(user_id)
        for i in range(30):
            await uow.conversations.add_message(
                convo.id, role=MessageRole.USER, content=f"msg-{i:02d}"
            )
        await uow.commit()
        convo_id = convo.id

    async with uow:
        recent = await uow.conversations.list_messages(convo_id, limit=10)
        assert [m.content for m in recent] == [f"msg-{i:02d}" for i in range(20, 30)]
        assert recent[-1].content == "msg-29"


@pytest.mark.asyncio
async def test_watchlist_add_get_deactivate(uow, demo_user):
    user_id = demo_user["user_id"]

    async with uow:
        item = await uow.watchlist.add(user_id, symbol="NVDA", name="NVIDIA", sector="Semis")
        await uow.commit()

        fetched = await uow.watchlist.get_by_symbol(user_id, "nvda")
        assert fetched is not None and fetched.id == item.id

    async with uow:
        items = await uow.watchlist.list_active(user_id)
        assert len(items) == 1

    async with uow:
        fetched = await uow.watchlist.get_by_symbol(user_id, "NVDA")
        await uow.watchlist.deactivate(fetched)
        await uow.commit()

    async with uow:
        assert await uow.watchlist.get_by_symbol(user_id, "NVDA") is None
        assert len(await uow.watchlist.list_active(user_id)) == 0


@pytest.mark.asyncio
async def test_watchlist_unique_symbol_per_user(uow, demo_user, session_factory):
    user_id = demo_user["user_id"]
    from sqlalchemy.exc import IntegrityError

    async with uow:
        await uow.watchlist.add(user_id, symbol="TSLA", name="Tesla", sector=None)
        await uow.commit()
    with pytest.raises(IntegrityError):
        async with uow:
            await uow.watchlist.add(user_id, symbol="TSLA", name="Tesla", sector=None)
            await uow.commit()


@pytest.mark.asyncio
async def test_watchlist_isolated_between_users(uow, demo_user, session_factory):
    async with uow:
        other = await uow.users.create(telegram_id=777, username="other")
        await uow.commit()
        other_id = other.id

    async with uow:
        await uow.watchlist.add(other_id, symbol="AAPL", name="Apple", sector=None)
        await uow.commit()

    async with uow:
        assert len(await uow.watchlist.list_active(demo_user["user_id"])) == 0
        assert len(await uow.watchlist.list_active(other_id)) == 1
