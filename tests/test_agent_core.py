"""AgentCore turn-loop tests: direct reply, tool loop, exhaustion, errors."""

import json

import pytest

from app.application.agent.core import AgentCore
from app.application.agent.ports import LLMResponse, LLMToolCall
from app.application.agent.tools import ToolContext, default_registry
from app.infrastructure.llm.gateway import LLMGatewayTransientError


class FakeGateway:
    """Scripted gateway: returns canned responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self.request_tools = []

    async def complete(self, messages, *, tools=None, max_tokens=600, temperature=0.3):
        self.requests.append(messages)
        self.request_tools.append(tools)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_agent(gateway, **kwargs) -> AgentCore:
    return AgentCore(gateway, default_registry(), max_tool_rounds=2, **kwargs)


@pytest.mark.asyncio
async def test_direct_reply_no_tools(uow, demo_user):
    async with uow:
        conversation = await uow.conversations.create(demo_user["user_id"])
        await uow.conversations.add_message(
            conversation.id, role="user", content="hi", content_type="text"
        )
        await uow.commit()
        conversation_id = conversation.id

    gateway = FakeGateway([LLMResponse(content="Hello!")])
    agent = make_agent(gateway)
    async with uow:
        reply = await agent.run(
            uow,
            user_id=demo_user["user_id"],
            conversation_id=conversation_id,
            tool_context=ToolContext(uow=uow, user_id=demo_user["user_id"]),
        )
    assert reply == "Hello!"


@pytest.mark.asyncio
async def test_tool_call_then_final_answer(uow, session_factory, demo_user):
    async with uow:
        conversation = await uow.conversations.create(demo_user["user_id"])
        await uow.conversations.add_message(
            conversation.id, role="user", content="quote aapl", content_type="text"
        )
        await uow.commit()
        conversation_id = conversation.id

    # First response requests a tool call; second produces the final answer.
    gateway = FakeGateway(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    LLMToolCall(
                        id="t_1",
                        name="get_market_quote",
                        arguments={"symbol": "AAPL"},
                    )
                ],
            ),
            LLMResponse(content="AAPL is at $100.00 today."),
        ]
    )
    agent = make_agent(gateway)

    async with uow:
        reply = await agent.run(
            uow,
            user_id=demo_user["user_id"],
            conversation_id=conversation_id,
            tool_context=ToolContext(uow=uow, user_id=demo_user["user_id"]),
        )
    assert reply == "AAPL is at $100.00 today."

    # The tool result was fed back to the model as a tool message.
    tool_messages = [m for m in gateway.requests[1] if m["role"] == "tool"]
    assert tool_messages, "tool result must be fed back"
    assert tool_messages[0]["tool_call_id"] == "t_1"


@pytest.mark.asyncio
async def test_max_tool_rounds_exhausted_returns_fallback(uow, demo_user):
    async with uow:
        conversation = await uow.conversations.create(demo_user["user_id"])
        await uow.commit()
        conversation_id = conversation.id

    gateway = FakeGateway(
        [
            LLMResponse(
                content=None, tool_calls=[LLMToolCall(id="t", name="list_memories", arguments={})]
            ),
            LLMResponse(
                content=None, tool_calls=[LLMToolCall(id="t", name="list_memories", arguments={})]
            ),
            LLMResponse(
                content=None, tool_calls=[LLMToolCall(id="t", name="list_memories", arguments={})]
            ),
            LLMResponse(content="Here is my final answer.", tool_calls=[]),
        ]
    )
    agent = make_agent(gateway, fallback_reply="Sorry, try again.")  # max_tool_rounds=2
    async with uow:
        reply = await agent.run(
            uow,
            user_id=demo_user["user_id"],
            conversation_id=conversation_id,
            tool_context=ToolContext(uow=uow, user_id=demo_user["user_id"]),
        )
    # Budget exhausted -> the caller gets one tools-free final pass, not the
    # generic fallback, so the user still receives a real answer.
    assert reply == "Here is my final answer."
    assert gateway.request_tools[-1] is None


@pytest.mark.asyncio
async def test_final_plain_reply_empty_still_falls_back(uow, demo_user):
    """If the tools-free final pass yields nothing, degrade to the fallback."""
    async with uow:
        conversation = await uow.conversations.create(demo_user["user_id"])
        await uow.commit()
        conversation_id = conversation.id

    gateway = FakeGateway(
        [
            LLMResponse(
                content=None, tool_calls=[LLMToolCall(id="t", name="list_memories", arguments={})]
            ),
            LLMResponse(
                content=None, tool_calls=[LLMToolCall(id="t", name="list_memories", arguments={})]
            ),
            LLMResponse(
                content=None, tool_calls=[LLMToolCall(id="t", name="list_memories", arguments={})]
            ),
            LLMResponse(content="", tool_calls=[]),
        ]
    )
    agent = make_agent(gateway, fallback_reply="Sorry, try again.")  # max_tool_rounds=2
    async with uow:
        reply = await agent.run(
            uow,
            user_id=demo_user["user_id"],
            conversation_id=conversation_id,
            tool_context=ToolContext(uow=uow, user_id=demo_user["user_id"]),
        )
    assert reply == "Sorry, try again."


@pytest.mark.asyncio
async def test_tool_exception_falls_back(uow, demo_user):
    """A non-LLM error mid-turn must NOT leave the user silent."""
    from app.application.agent.tools import Tool, ToolRegistry

    async def _explode(ctx, args):
        raise RuntimeError("provider exploded")

    boom_tool = Tool(name="boom", description="explodes", parameters={}, handler=_explode)
    registry = ToolRegistry([boom_tool])

    async with uow:
        conversation = await uow.conversations.create(demo_user["user_id"])
        await uow.commit()
        conversation_id = conversation.id

    gateway = FakeGateway(
        [
            LLMResponse(content=None, tool_calls=[LLMToolCall(id="t", name="boom", arguments={})]),
            LLMResponse(content="recovery text"),
        ]
    )
    agent = AgentCore(gateway, registry, max_tool_rounds=2, fallback_reply="Sorry, try again.")
    async with uow:
        reply = await agent.run(
            uow,
            user_id=demo_user["user_id"],
            conversation_id=conversation_id,
            tool_context=ToolContext(uow=uow, user_id=demo_user["user_id"]),
        )
    assert reply == "Sorry, try again."


@pytest.mark.asyncio
async def test_gateway_error_returns_fallback(uow, demo_user):
    async with uow:
        conversation = await uow.conversations.create(demo_user["user_id"])
        await uow.commit()
        conversation_id = conversation.id

    gateway = FakeGateway([LLMGatewayTransientError("boom")])
    agent = make_agent(gateway, fallback_reply="Sorry, try again.")
    async with uow:
        reply = await agent.run(
            uow,
            user_id=demo_user["user_id"],
            conversation_id=conversation_id,
            tool_context=ToolContext(uow=uow, user_id=demo_user["user_id"]),
        )
    assert reply == "Sorry, try again."


@pytest.mark.asyncio
async def test_empty_reply_returns_fallback(uow, demo_user):
    async with uow:
        conversation = await uow.conversations.create(demo_user["user_id"])
        await uow.commit()
        conversation_id = conversation.id

    gateway = FakeGateway([LLMResponse(content="   ")])
    agent = make_agent(gateway, fallback_reply="Sorry, try again.")
    async with uow:
        reply = await agent.run(
            uow,
            user_id=demo_user["user_id"],
            conversation_id=conversation_id,
            tool_context=ToolContext(uow=uow, user_id=demo_user["user_id"]),
        )
    assert reply == "Sorry, try again."


@pytest.mark.asyncio
async def test_market_turn_without_tool_forces_retry(uow, demo_user):
    """A market question answered with zero tool calls gets one re-prompt so
    the model cannot fabricate prices."""
    async with uow:
        conversation = await uow.conversations.create(demo_user["user_id"])
        await uow.conversations.add_message(
            conversation.id, role="user", content="What is NVDA trading at?", content_type="text"
        )
        await uow.commit()
        conversation_id = conversation.id

    gateway = FakeGateway(
        [
            LLMResponse(content="NVDA is at $125.50."),  # invented, no tool called
            LLMResponse(
                content=None,
                tool_calls=[
                    LLMToolCall(
                        id="t_2", name="get_market_quote", arguments={"symbol": "NVDA"}
                    )
                ],
            ),
            LLMResponse(content="NVDA is at $222.47."),
        ]
    )
    agent = make_agent(gateway)
    async with uow:
        reply = await agent.run(
            uow,
            user_id=demo_user["user_id"],
            conversation_id=conversation_id,
            tool_context=ToolContext(uow=uow, user_id=demo_user["user_id"]),
        )
    assert reply == "NVDA is at $222.47."
    assert len(gateway.requests) == 3
    forced = [
        m
        for m in gateway.requests[1]
        if m["role"] == "user" and "Never invent" in m["content"]
    ]
    assert forced, "market retry message must be appended before the tool round"


@pytest.mark.asyncio
async def test_market_turn_with_tool_call_not_retried(uow, demo_user):
    async with uow:
        conversation = await uow.conversations.create(demo_user["user_id"])
        await uow.conversations.add_message(
            conversation.id, role="user", content="What is NVDA trading at?", content_type="text"
        )
        await uow.commit()
        conversation_id = conversation.id

    gateway = FakeGateway(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    LLMToolCall(
                        id="t_3", name="get_market_quote", arguments={"symbol": "NVDA"}
                    )
                ],
            ),
            LLMResponse(content="NVDA is at $222.47."),
        ]
    )
    agent = make_agent(gateway)
    async with uow:
        reply = await agent.run(
            uow,
            user_id=demo_user["user_id"],
            conversation_id=conversation_id,
            tool_context=ToolContext(uow=uow, user_id=demo_user["user_id"]),
        )
    assert reply == "NVDA is at $222.47."
    assert len(gateway.requests) == 2  # no forced re-prompt when tools were used


@pytest.mark.asyncio
async def test_tool_execution_feeds_valid_json(uow, demo_user):
    """The serialized tool_calls appended to messages must be loadable JSON."""
    async with uow:
        conversation = await uow.conversations.create(demo_user["user_id"])
        await uow.commit()
        conversation_id = conversation.id

    gateway = FakeGateway(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    LLMToolCall(
                        id="t", name="save_memory", arguments={"memory_key": "k", "summary": "s"}
                    )
                ],
            ),
            LLMResponse(content="done"),
        ]
    )
    agent = make_agent(gateway)
    async with uow:
        await agent.run(
            uow,
            user_id=demo_user["user_id"],
            conversation_id=conversation_id,
            tool_context=ToolContext(uow=uow, user_id=demo_user["user_id"]),
        )
    assistant_message = gateway.requests[1][-2]
    assert assistant_message["role"] == "assistant"
    args = assistant_message["tool_calls"][0]["function"]["arguments"]
    assert json.loads(args)["memory_key"] == "k"

    async with uow:
        memories = await uow.memories.list_active(demo_user["user_id"])
        assert any(m.memory_key == "k" for m in memories)
