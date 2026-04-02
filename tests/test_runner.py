"""Tests for AgentRunner with mock LLM client."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from marvin.context import CustomerContextBundle
from marvin.llm.base import LLMMessage, LLMResponse, StopReason
from marvin.models import AgentConfig, LLMProvider
from marvin.runner import AgentRunner, _build_context_prefix
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


def _make_bundle(**kwargs: Any) -> CustomerContextBundle:
    defaults = {
        "customer_id": "acme",
        "claude_md": "# Claude Config\n\nBe helpful.",
        "sops": {"deploy.md": "# Deploy SOP\n\nStep 1: push."},
        "project_goals": "# Goals\n\nShip fast.",
        "memory_index": "",
        "daily_memories": [],
    }
    defaults.update(kwargs)
    return CustomerContextBundle(**defaults)


class TestBuildContextPrefix:
    def test_includes_claude_md(self) -> None:
        bundle = _make_bundle(sops={}, project_goals="")
        prefix = _build_context_prefix(bundle)
        assert "# CLAUDE.md" in prefix
        assert "Be helpful." in prefix

    def test_includes_each_sop_file(self) -> None:
        bundle = _make_bundle(sops={"a.md": "Content A", "b.md": "Content B"})
        prefix = _build_context_prefix(bundle)
        assert "## a.md" in prefix
        assert "Content A" in prefix
        assert "## b.md" in prefix
        assert "Content B" in prefix

    def test_includes_project_goals(self) -> None:
        bundle = _make_bundle()
        prefix = _build_context_prefix(bundle)
        assert "# Project Goals" in prefix
        assert "Ship fast." in prefix

    def test_empty_fields_omitted(self) -> None:
        bundle = _make_bundle(claude_md="", sops={}, project_goals="")
        prefix = _build_context_prefix(bundle)
        assert prefix == ""


class TestAgentRunnerContextBundle:
    def _patch_client(self, runner: AgentRunner, response_content: str = "OK") -> MagicMock:
        mock_client = MagicMock()
        mock_response = _make_response(response_content)
        mock_client.generate = AsyncMock(return_value=mock_response)
        mock_client.generate_with_tools = AsyncMock(
            return_value=(mock_response, [LLMMessage.user("Hi"), LLMMessage.assistant(response_content)])
        )
        runner._client = mock_client
        return mock_client

    def test_with_bundle_prefix_prepended_to_system_prompt(self) -> None:
        agent = _make_agent(system_prompt="You are an assistant.")
        bundle = _make_bundle()
        runner = AgentRunner(agent=agent, register_builtins=False, context_bundle=bundle)
        mock_client = self._patch_client(runner)

        asyncio.run(runner.chat("Hi", enable_tools=False))

        call_kwargs = mock_client.generate.call_args[1]
        prompt = call_kwargs.get("system_prompt", "")
        assert "# CLAUDE.md" in prompt
        assert "# Project Goals" in prompt
        assert "You are an assistant." in prompt
        # prefix comes before the original prompt
        assert prompt.index("# CLAUDE.md") < prompt.index("You are an assistant.")

    def test_without_bundle_no_regression(self) -> None:
        agent = _make_agent(system_prompt="You are an assistant.")
        runner = AgentRunner(agent=agent, register_builtins=False)
        mock_client = self._patch_client(runner)

        asyncio.run(runner.chat("Hi", enable_tools=False))

        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs.get("system_prompt") == "You are an assistant."

    def test_with_bundle_and_explicit_system_prompt_combined(self) -> None:
        agent = _make_agent(system_prompt="Agent default.")
        bundle = _make_bundle()
        runner = AgentRunner(agent=agent, register_builtins=False, context_bundle=bundle)
        mock_client = self._patch_client(runner)

        asyncio.run(runner.chat("Hi", system_prompt="Explicit override.", enable_tools=False))

        call_kwargs = mock_client.generate.call_args[1]
        prompt = call_kwargs.get("system_prompt", "")
        assert "# CLAUDE.md" in prompt
        assert "Explicit override." in prompt
        # prefix before the explicit override
        assert prompt.index("# CLAUDE.md") < prompt.index("Explicit override.")

    def test_with_bundle_no_system_prompt_uses_prefix_only(self) -> None:
        agent = _make_agent()
        bundle = _make_bundle(sops={}, project_goals="")
        runner = AgentRunner(agent=agent, register_builtins=False, context_bundle=bundle)
        mock_client = self._patch_client(runner)

        asyncio.run(runner.run([LLMMessage.user("Hi")], enable_tools=False))

        call_kwargs = mock_client.generate.call_args[1]
        prompt = call_kwargs.get("system_prompt", "")
        assert "# CLAUDE.md" in prompt
