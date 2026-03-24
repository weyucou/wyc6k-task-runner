"""Tests for AgentRunner with mock LLM client."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from marvin.llm.base import LLMMessage, LLMResponse, StopReason
from marvin.models import AgentConfig, LLMProvider, ToolProfile
from marvin.runner import AgentRunner
from marvin.tools import ToolRegistry
from marvin.tools.base import ToolResult


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

        asyncio.run(
            runner.chat("Hi", system_prompt="Override prompt.", enable_tools=False)
        )

        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs.get("system_prompt") == "Override prompt."

    def test_rate_limit_not_applied_when_disabled(self) -> None:
        agent = _make_agent(rate_limit_enabled=False)
        runner = AgentRunner(agent=agent, register_builtins=False)
        self._patch_client(runner)

        # Should complete without blocking
        asyncio.run(runner.chat("Hi", enable_tools=False))
