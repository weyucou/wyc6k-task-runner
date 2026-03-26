"""Tests for AgentRunner with mock LLM client."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from marvin.llm.base import LLMMessage, LLMResponse, StopReason
from marvin.models import AgentConfig, LLMProvider
from marvin.runner import AgentRunner, history_to_llm_messages
from marvin.tools import ToolRegistry


def _make_agent(**kwargs: Any) -> AgentConfig:
    defaults = {
        "name": "test-agent",
        "provider": LLMProvider.ANTHROPIC,
        "model_name": "claude-sonnet-4-20250514",
    }
    defaults.update(kwargs)
    return AgentConfig(**defaults)


def _make_response(content: str = "Hello!") -> LLMResponse:
    return LLMResponse(content=content, stop_reason=StopReason.END_TURN)


class TestAgentRunnerInit:
    def test_creates_default_registry(self) -> None:
        agent = _make_agent()
        runner = AgentRunner(agent=agent, register_builtins=False)
        assert runner.registry is not None
        assert runner.agent is agent

    def test_accepts_custom_registry(self) -> None:
        registry = ToolRegistry()
        runner = AgentRunner(register_builtins=False, registry=registry)
        assert runner.registry is registry

    def test_no_agent_configured(self) -> None:
        runner = AgentRunner(register_builtins=False)
        with pytest.raises(ValueError, match="No agent configured"):
            runner.get_client()

    def test_register_builtins_populates_tools(self) -> None:
        agent = _make_agent()
        runner = AgentRunner(agent=agent, register_builtins=True)
        # Should have core + coding tools registered
        assert len(runner.registry.list_tools()) > 0
        assert "calculator" in runner.registry.list_tools()
        assert "read" in runner.registry.list_tools()


class TestAgentRunnerGetTools:
    def test_anthropic_format(self) -> None:
        agent = _make_agent(provider=LLMProvider.ANTHROPIC)
        runner = AgentRunner(agent=agent, register_builtins=False)
        from marvin.tools.builtin import CalculatorTool

        runner.register_tool(CalculatorTool())
        tools = runner.get_tools_for_provider("anthropic")
        assert len(tools) == 1
        assert "input_schema" in tools[0]

    def test_openai_format(self) -> None:
        agent = _make_agent(provider=LLMProvider.ANTHROPIC)
        runner = AgentRunner(agent=agent, register_builtins=False)
        from marvin.tools.builtin import CalculatorTool

        runner.register_tool(CalculatorTool())
        tools = runner.get_tools_for_provider("openai")
        assert len(tools) == 1
        assert tools[0]["type"] == "function"

    def test_vllm_uses_openai_format(self) -> None:
        agent = _make_agent()
        runner = AgentRunner(agent=agent, register_builtins=False)
        from marvin.tools.builtin import CalculatorTool

        runner.register_tool(CalculatorTool())
        tools = runner.get_tools_for_provider("vllm")
        assert tools[0]["type"] == "function"

    def test_ollama_uses_openai_format(self) -> None:
        agent = _make_agent()
        runner = AgentRunner(agent=agent, register_builtins=False)
        from marvin.tools.builtin import CalculatorTool

        runner.register_tool(CalculatorTool())
        tools = runner.get_tools_for_provider("ollama")
        assert tools[0]["type"] == "function"


class TestAgentRunnerChat:
    def _patch_client(self, runner: AgentRunner, response_content: str = "Test response") -> MagicMock:
        """Inject a mock LLM client into the runner."""
        mock_client = MagicMock()
        mock_response = _make_response(response_content)
        mock_client.generate = AsyncMock(return_value=mock_response)
        mock_client.generate_with_tools = AsyncMock(
            return_value=(mock_response, [LLMMessage.user("Hi"), LLMMessage.assistant(response_content)])
        )
        runner._client = mock_client
        return mock_client

    def test_chat_returns_response_text(self) -> None:
        agent = _make_agent()
        runner = AgentRunner(agent=agent, register_builtins=False)
        self._patch_client(runner, "Hello from agent!")

        text, history = asyncio.run(runner.chat("Hi", enable_tools=False))
        assert text == "Hello from agent!"
        assert len(history) >= 1

    def test_chat_appends_user_message(self) -> None:
        agent = _make_agent()
        runner = AgentRunner(agent=agent, register_builtins=False)
        mock_client = self._patch_client(runner, "Reply")

        asyncio.run(runner.chat("My question", enable_tools=False))

        # generate was called with messages including our user message
        call_args = mock_client.generate.call_args
        messages_passed = call_args[0][0]
        assert any(m.content == "My question" for m in messages_passed)

    def test_chat_uses_system_prompt_from_agent(self) -> None:
        agent = _make_agent(system_prompt="You are helpful.")
        runner = AgentRunner(agent=agent, register_builtins=False)
        mock_client = self._patch_client(runner)

        asyncio.run(runner.chat("Hi", enable_tools=False))

        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs.get("system_prompt") == "You are helpful."

    def test_chat_system_prompt_override(self) -> None:
        agent = _make_agent(system_prompt="Original prompt.")
        runner = AgentRunner(agent=agent, register_builtins=False)
        mock_client = self._patch_client(runner)

        asyncio.run(runner.chat("Hi", system_prompt="Override prompt.", enable_tools=False))

        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs.get("system_prompt") == "Override prompt."

    def test_rate_limit_not_applied_when_disabled(self) -> None:
        agent = _make_agent(rate_limit_enabled=False)
        runner = AgentRunner(agent=agent, register_builtins=False)
        self._patch_client(runner)

        # Should complete without blocking
        asyncio.run(runner.chat("Hi", enable_tools=False))


class TestHistoryToLlmMessages:
    def test_user_message(self) -> None:
        records = [{"role": "user", "content": "Hello"}]
        result = history_to_llm_messages(records)
        assert len(result) == 1
        assert result[0].role.value == "user"
        assert result[0].content == "Hello"

    def test_assistant_message(self) -> None:
        records = [{"role": "assistant", "content": "Hi there"}]
        result = history_to_llm_messages(records)
        assert len(result) == 1
        assert result[0].role.value == "assistant"
        assert result[0].content == "Hi there"
        assert result[0].tool_calls is None

    def test_assistant_message_with_tool_calls(self) -> None:
        records = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc1", "name": "calculator", "arguments": {"expression": "1+1"}}],
            }
        ]
        result = history_to_llm_messages(records)
        assert len(result) == 1
        assert result[0].tool_calls is not None
        assert len(result[0].tool_calls) == 1
        tc = result[0].tool_calls[0]
        assert tc.id == "tc1"
        assert tc.name == "calculator"
        assert tc.arguments == {"expression": "1+1"}

    def test_tool_result_message(self) -> None:
        records = [{"role": "tool", "content": "2", "tool_call_id": "tc1", "name": "calculator"}]
        result = history_to_llm_messages(records)
        assert len(result) == 1
        assert result[0].role.value == "tool"
        assert result[0].content == "2"
        assert result[0].tool_call_id == "tc1"
        assert result[0].name == "calculator"

    def test_system_message(self) -> None:
        records = [{"role": "system", "content": "You are helpful."}]
        result = history_to_llm_messages(records)
        assert len(result) == 1
        assert result[0].role.value == "system"

    def test_unknown_role_falls_back_to_user(self) -> None:
        records = [{"role": "unknown", "content": "test"}]
        result = history_to_llm_messages(records)
        assert result[0].role.value == "user"

    def test_empty_records(self) -> None:
        assert history_to_llm_messages([]) == []

    def test_mixed_roles_preserves_order(self) -> None:
        records = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
            {"role": "tool", "content": "result", "tool_call_id": "x"},
        ]
        result = history_to_llm_messages(records)
        assert len(result) == 3
        assert result[0].role.value == "user"
        assert result[1].role.value == "assistant"
        assert result[2].role.value == "tool"


class TestContextWindowMessages:
    def _patch_client(self, runner: AgentRunner, response_content: str = "Reply") -> MagicMock:
        mock_client = MagicMock()
        mock_response = _make_response(response_content)
        mock_client.generate = AsyncMock(return_value=mock_response)
        mock_client.generate_with_tools = AsyncMock(
            return_value=(mock_response, [LLMMessage.user("Hi"), LLMMessage.assistant(response_content)])
        )
        runner._client = mock_client
        return mock_client

    def _get_input_messages(self, mock_client: MagicMock) -> list[LLMMessage]:
        """Extract the messages list passed to generate, excluding any post-call appends.

        When enable_tools=False, run() sets history=messages (same reference), so
        chat()'s append mutates the list visible in call_args. We slice off the last
        element (the assistant message appended after generate returns) to get the
        exact list that was passed in.
        """
        raw: list[LLMMessage] = mock_client.generate.call_args[0][0]
        return raw[:-1]  # exclude the assistant message appended by chat()

    def test_zero_context_window_passes_full_history(self) -> None:
        agent = _make_agent(context_window_messages=0)
        runner = AgentRunner(agent=agent, register_builtins=False)
        mock_client = self._patch_client(runner)

        history = [LLMMessage.user(f"msg{i}") for i in range(5)]
        asyncio.run(runner.chat("new", conversation_history=history, enable_tools=False))

        messages_passed = self._get_input_messages(mock_client)
        # All 5 history messages + the new user message = 6
        assert len(messages_passed) == 6

    def test_context_window_limits_injected_messages(self) -> None:
        agent = _make_agent(context_window_messages=2)
        runner = AgentRunner(agent=agent, register_builtins=False)
        mock_client = self._patch_client(runner)

        history = [LLMMessage.user(f"msg{i}") for i in range(5)]
        asyncio.run(runner.chat("new", conversation_history=history, enable_tools=False))

        messages_passed = self._get_input_messages(mock_client)
        # Last 2 history messages + the new user message = 3
        assert len(messages_passed) == 3
        assert messages_passed[0].content == "msg3"
        assert messages_passed[1].content == "msg4"
        assert messages_passed[2].content == "new"

    def test_context_window_larger_than_history_uses_full_history(self) -> None:
        agent = _make_agent(context_window_messages=10)
        runner = AgentRunner(agent=agent, register_builtins=False)
        mock_client = self._patch_client(runner)

        history = [LLMMessage.user(f"msg{i}") for i in range(3)]
        asyncio.run(runner.chat("new", conversation_history=history, enable_tools=False))

        messages_passed = self._get_input_messages(mock_client)
        # All 3 + new = 4
        assert len(messages_passed) == 4

    def test_context_window_with_no_history(self) -> None:
        agent = _make_agent(context_window_messages=5)
        runner = AgentRunner(agent=agent, register_builtins=False)
        mock_client = self._patch_client(runner)

        asyncio.run(runner.chat("first message", enable_tools=False))

        messages_passed = self._get_input_messages(mock_client)
        assert len(messages_passed) == 1
        assert messages_passed[0].content == "first message"
