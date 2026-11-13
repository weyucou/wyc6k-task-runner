# Copyright (c) 2026 MNKT
"""Tests for AgentRunner with mock LLM client."""

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from marvin.context import CustomerContextBundle
from marvin.llm.base import LLMMessage, LLMResponse, MessageRole, StopReason, ToolCall
from marvin.models import AgentConfig, LLMProvider, ToolProfile
from marvin.runner import AgentRunner, _build_context_prefix
from marvin.tools import ToolRegistry
from marvin.tools.base import BaseTool
from marvin.tools.builtin import CalculatorTool
from tests.fake_llm import (
    FakeLLMClient,
    FakeRequest,
    RecordingTool,
    fake_text,
    fake_tool_call,
    fake_tool_calls,
)

# The `coding` profile's tools that are registered: every built-in tool except `browser`, `calculator`
# and `get_datetime`.
CODING_TOOLS = {
    "apply_patch",
    "askcc_run",
    "browser_fetch",
    "edit",
    "exec",
    "github_issue",
    "github_pr",
    "image",
    "memory_retrieve",
    "memory_search",
    "memory_store",
    "process",
    "read",
    "sessions_send",
    "sessions_spawn",
    "web_fetch",
    "web_search",
    "write",
}


def _make_agent(**kwargs: Any) -> AgentConfig:
    defaults = {
        "name": "test-agent",
        "provider": LLMProvider.ANTHROPIC,
        "model_name": "claude-sonnet-5-5",
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
        runner.register_tool(CalculatorTool())
        tools = runner.get_tools_for_provider("anthropic")
        assert len(tools) == 1
        assert "input_schema" in tools[0]

    def test_openai_format(self) -> None:
        agent = _make_agent(provider=LLMProvider.ANTHROPIC)
        runner = AgentRunner(agent=agent, register_builtins=False)
        runner.register_tool(CalculatorTool())
        tools = runner.get_tools_for_provider("openai")
        assert len(tools) == 1
        assert tools[0]["type"] == "function"

    def test_vllm_uses_openai_format(self) -> None:
        agent = _make_agent()
        runner = AgentRunner(agent=agent, register_builtins=False)
        runner.register_tool(CalculatorTool())
        tools = runner.get_tools_for_provider("vllm")
        assert tools[0]["type"] == "function"

    def test_ollama_uses_openai_format(self) -> None:
        agent = _make_agent()
        runner = AgentRunner(agent=agent, register_builtins=False)
        runner.register_tool(CalculatorTool())
        tools = runner.get_tools_for_provider("ollama")
        assert tools[0]["type"] == "function"


class TestAgentRunnerChat:
    def _patch_client(self, runner: AgentRunner, response_content: str = "Test response") -> MagicMock:
        """Inject a mock LLM client into the runner."""
        mock_client = MagicMock()
        mock_response = _make_response(response_content)
        mock_client.generate = AsyncMock(return_value=mock_response)
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


class TestAgentRunnerToolLoop:
    def test_run_drives_the_tool_loop_with_agent_settings(self) -> None:
        agent = _make_agent(system_prompt="You are helpful.", temperature=0.2, max_tokens=512)
        runner = AgentRunner(agent=agent, register_builtins=False)
        runner.register_tool(CalculatorTool())
        client = FakeLLMClient(
            [fake_tool_calls(fake_tool_call("calculator", {"expression": "2 + 2"})), fake_text("2 + 2 = 4")]
        )
        runner._client = client

        response, history = asyncio.run(
            runner.run([LLMMessage.user("What is 2 + 2?")], enable_tools=True, max_tool_iterations=1)
        )

        assert response.content == "2 + 2 = 4"
        assert [m.role for m in history] == [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL]
        assert history[-1].content == "4"  # CalculatorTool ran through runner.registry
        tools = runner.registry.to_anthropic_tools()
        # max_tool_iterations=1 makes the second call the capped final call, which has no tools.
        assert [(r.system_prompt, r.tools, r.temperature, r.max_tokens) for r in client.requests] == [
            ("You are helpful.", tools, 0.2, 512),
            ("You are helpful.", None, 0.2, 512),
        ]


def _run_turns(
    runner: AgentRunner, *responses: LLMResponse, **run_kwargs: Any
) -> tuple[FakeLLMClient, list[LLMMessage]]:
    """Run one user message through `runner` against scripted `responses`; return the client and the history."""
    client = FakeLLMClient(list(responses))
    runner._client = client
    _, history = asyncio.run(runner.run([LLMMessage.user("Go.")], **run_kwargs))
    return client, history


def _sent_tool_names(request: FakeRequest) -> set[str]:
    return {tool["name"] for tool in request.tools or []}


def _runner_with(*tools: BaseTool, agent: AgentConfig, **kwargs: Any) -> AgentRunner:
    """A runner whose registry holds only `tools`."""
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return AgentRunner(agent=agent, registry=registry, register_builtins=False, **kwargs)


class TestAgentRunnerToolFiltering:
    def test_minimal_profile_sends_no_tools(self) -> None:
        runner = AgentRunner(agent=_make_agent(tool_profile=ToolProfile.MINIMAL))

        client, _ = _run_turns(runner, fake_text("Hi."))

        assert [request.tools for request in client.requests] == [None]

    @pytest.mark.parametrize(
        ("agent_kwargs", "expected"),
        [
            pytest.param({"tool_profile": ToolProfile.CODING}, CODING_TOOLS, id="coding"),
            pytest.param(
                {"tool_profile": ToolProfile.MINIMAL, "tools_allow": ["get_datetime"]},
                {"get_datetime"},
                id="minimal-plus-allow",
            ),
        ],
    )
    def test_sent_tools_follow_the_profile(self, agent_kwargs: dict[str, Any], expected: set[str]) -> None:
        runner = AgentRunner(agent=_make_agent(**agent_kwargs))

        client, _ = _run_turns(runner, fake_text("Hi."))

        assert _sent_tool_names(client.requests[0]) == expected

    def test_denied_tool_is_not_sent(self) -> None:
        runner = AgentRunner(agent=_make_agent(tool_profile=ToolProfile.FULL, tools_deny=["exec"]))

        client, _ = _run_turns(runner, fake_text("Hi."))

        assert _sent_tool_names(client.requests[0]) == set(runner.registry.list_tools()) - {"exec"}

    def test_tool_names_narrow_the_allowed_tools(self) -> None:
        calculator, exec_tool = RecordingTool("calculator"), RecordingTool("exec")
        runner = _runner_with(calculator, exec_tool, agent=_make_agent())

        client, history = _run_turns(
            runner,
            fake_tool_calls(fake_tool_call("exec", {})),
            fake_text("OK."),
            tool_names=["calculator"],
        )

        assert _sent_tool_names(client.requests[0]) == {"calculator"}
        assert exec_tool.calls == []
        assert history[-1].content.startswith("Error: Tool 'exec' is not available")


class TestAgentRunnerToolPolicy:
    def test_call_to_a_denied_tool_is_rejected(self) -> None:
        exec_tool, read_tool = RecordingTool("exec"), RecordingTool("read")
        runner = _runner_with(exec_tool, read_tool, agent=_make_agent(tools_deny=["exec"]))

        _, history = _run_turns(runner, fake_tool_calls(fake_tool_call("exec", {"command": "id"})), fake_text("OK."))

        assert exec_tool.calls == []
        assert history[-1].content == (
            "Error: Tool 'exec' is not available to this agent (profile: full). Available tools: read."
        )

    def test_caller_hook_rejection_reaches_the_model(self) -> None:
        seen: list[str] = []

        async def pause_read(call: ToolCall) -> str | None:
            seen.append(call.name)
            return "read is paused for this customer"

        exec_tool, read_tool = RecordingTool("exec"), RecordingTool("read")
        runner = _runner_with(exec_tool, read_tool, agent=_make_agent(tools_deny=["exec"]), before_tool_call=pause_read)

        _, history = _run_turns(
            runner,
            fake_tool_calls(fake_tool_call("exec", {}), fake_tool_call("read", {})),
            fake_text("OK."),
        )

        assert (exec_tool.calls, read_tool.calls) == ([], [])
        assert seen == ["read"]  # the policy rejected exec before the caller's hook ran
        assert history[-1].content == "Error: read is paused for this customer"

    def test_require_approval_tool_runs_and_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        write_tool = RecordingTool("write", require_approval=True)
        runner = _runner_with(write_tool, agent=_make_agent())

        with caplog.at_level(logging.INFO, logger="marvin.tools.policy"):
            _, history = _run_turns(runner, fake_tool_calls(fake_tool_call("write", {"path": "a"})), fake_text("OK."))

        assert write_tool.calls == [{"path": "a"}]
        assert history[-1].content == "write ran"
        assert "auto-approved write (unattended run)" in caplog.messages


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
