# Copyright (c) 2026 MNKT
"""Tests for the shared tool-call loop, driven by the scripted FakeLLMClient."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from marvin.llm.base import LLMMessage, LLMResponse, MessageRole
from marvin.loop import run_tool_loop
from marvin.tools import ToolRegistry
from marvin.tools.base import BaseTool, ToolResult
from marvin.tools.builtin import CalculatorTool
from marvin.tools.coding import ExecTool
from tests.fake_llm import (
    FakeLLMClient,
    fake_error,
    fake_max_tokens,
    fake_text,
    fake_tool_call,
    fake_tool_calls,
)

QUESTION = LLMMessage.user("What is 2 + 2?")


class RaisingTool(BaseTool):
    """Raises inside `execute`, which `ToolRegistry.execute` catches."""

    name = "raising"
    description = "Always raises."
    parameters = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("tool exploded")


class BadSchemaTool(BaseTool):
    """Raises in `validate_params`, before the tool body runs."""

    name = "bad_schema"
    description = "Parameter validation raises."
    parameters = []

    def validate_params(self, params: dict[str, Any]) -> tuple[bool, str | None]:
        raise ValueError("schema exploded")

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult.success("unreachable")


@pytest.fixture
def registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(RaisingTool())
    registry.register(BadSchemaTool())
    return registry


async def _run_one_call(registry: ToolRegistry, name: str, arguments: dict[str, Any]) -> LLMMessage:
    """Run a loop whose model calls one tool, then answers; return the tool result message."""
    call = fake_tool_call(name, arguments)
    client = FakeLLMClient([fake_tool_calls(call), fake_text("done")])

    response, history = await run_tool_loop(client, [QUESTION], registry.to_anthropic_tools(), registry)

    assert response.content == "done"
    assert len(client.requests) == 2
    tool_result = history[-1]
    assert (tool_result.role, tool_result.tool_call_id, tool_result.name) == (MessageRole.TOOL, call.id, name)
    return tool_result


class TestRunToolLoop:
    @pytest.mark.asyncio
    async def test_final_answer_without_tool_calls(self, registry: ToolRegistry) -> None:
        answer = fake_text("4")
        client = FakeLLMClient([answer])
        messages = [QUESTION]
        tools = registry.to_anthropic_tools()

        response, history = await run_tool_loop(client, messages, tools, registry)

        assert response is answer
        assert len(client.requests) == 1
        assert client.requests[0].messages == [QUESTION]
        assert client.requests[0].tools == tools
        assert history == [QUESTION]
        assert history is not messages
        assert messages == [QUESTION]

    @pytest.mark.asyncio
    async def test_one_tool_round_trip(self, registry: ToolRegistry) -> None:
        call = fake_tool_call("calculator", {"expression": "2 + 2"})
        client = FakeLLMClient([fake_tool_calls(call, content="Let me check."), fake_text("2 + 2 = 4")])
        messages = [QUESTION]

        response, history = await run_tool_loop(client, messages, registry.to_anthropic_tools(), registry)

        assert response.content == "2 + 2 = 4"
        assert len(client.requests) == 2
        user, assistant, tool_result = client.requests[1].messages
        assert user == QUESTION
        assert (assistant.role, assistant.content, assistant.tool_calls) == (
            MessageRole.ASSISTANT,
            "Let me check.",
            [call],
        )
        assert (tool_result.role, tool_result.tool_call_id, tool_result.name, tool_result.content) == (
            MessageRole.TOOL,
            call.id,
            "calculator",
            "4",
        )
        assert not tool_result.is_error
        # The final answer is not appended to the history; AgentRunner.chat appends it.
        assert history == [user, assistant, tool_result]
        assert messages == [QUESTION]

    @pytest.mark.asyncio
    async def test_several_tool_calls_in_one_turn(self, registry: ToolRegistry) -> None:
        first = fake_tool_call("calculator", {"expression": "2 + 2"})
        second = fake_tool_call("calculator", {"expression": "3 * 3"})
        client = FakeLLMClient([fake_tool_calls(first, second), fake_text("4 and 9")])

        _, history = await run_tool_loop(client, [QUESTION], registry.to_anthropic_tools(), registry)

        assert len(client.requests) == 2
        assert [m.role for m in history] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.TOOL,
            MessageRole.TOOL,
        ]
        assert history[1].tool_calls == [first, second]
        assert [(m.tool_call_id, m.content) for m in history[2:]] == [(first.id, "4"), (second.id, "9")]

    @pytest.mark.asyncio
    async def test_unknown_tool_lists_the_available_tools(self, registry: ToolRegistry) -> None:
        tool_result = await _run_one_call(registry, "nope", {})

        assert tool_result.is_error
        assert tool_result.content == "Error: Tool 'nope' not found. Available tools: calculator, raising, bad_schema."

    @pytest.mark.asyncio
    async def test_invalid_arguments_ask_for_a_corrected_call(self, registry: ToolRegistry) -> None:
        tool_result = await _run_one_call(registry, "calculator", {})

        assert tool_result.is_error
        assert tool_result.content == (
            "Error: Invalid arguments for tool 'calculator': Missing required parameter: expression. "
            "Fix the arguments and call the tool again."
        )

    @pytest.mark.asyncio
    async def test_tool_exception_is_reported(self, registry: ToolRegistry) -> None:
        tool_result = await _run_one_call(registry, "raising", {})

        assert tool_result.is_error
        assert tool_result.content == "Error: tool exploded"

    @pytest.mark.asyncio
    async def test_error_outside_the_tool_body_is_reported(self, registry: ToolRegistry) -> None:
        tool_result = await _run_one_call(registry, "bad_schema", {})

        assert tool_result.is_error
        assert tool_result.content == "Error: schema exploded"

    @pytest.mark.asyncio
    async def test_registry_failure_is_reported(self, registry: ToolRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(registry, "execute", AsyncMock(side_effect=RuntimeError("registry exploded")))

        tool_result = await _run_one_call(registry, "calculator", {"expression": "2 + 2"})

        assert tool_result.is_error
        assert tool_result.content == "Error executing tool: registry exploded"

    @pytest.mark.asyncio
    async def test_non_zero_exit_reports_the_exit_code(self, registry: ToolRegistry) -> None:
        registry.register(ExecTool())

        tool_result = await _run_one_call(registry, "exec", {"command": "echo partial; exit 3"})

        # The partial output comes first, the error last.
        assert tool_result.is_error
        assert tool_result.content.startswith("partial\n")
        assert tool_result.content.endswith("Error: Command exited with code 3.")

    @pytest.mark.asyncio
    async def test_iteration_cap_ends_with_a_call_without_tools(self, registry: ToolRegistry) -> None:
        tools = registry.to_anthropic_tools()
        client = FakeLLMClient(
            [
                fake_tool_calls(fake_tool_call("calculator", {"expression": "1 + 1"})),
                fake_tool_calls(fake_tool_call("calculator", {"expression": "2 + 2"})),
                fake_text("Stopping here."),
            ]
        )

        response, history = await run_tool_loop(
            client,
            [QUESTION],
            tools,
            registry,
            system_prompt="Be brief.",
            temperature=0.2,
            max_tokens=256,
            max_iterations=2,
        )

        assert response.content == "Stopping here."
        assert [request.tools for request in client.requests] == [tools, tools, None]
        for request in client.requests:
            assert (request.system_prompt, request.temperature, request.max_tokens) == ("Be brief.", 0.2, 256)
        assert [m.content for m in history if m.role == MessageRole.TOOL] == ["2", "4"]
        assert client.requests[2].messages == history

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "scripted",
        [fake_error("rate limited"), fake_max_tokens("The answer is")],
        ids=["error", "max_tokens"],
    )
    async def test_error_and_max_tokens_stops_are_returned_unchanged(
        self, registry: ToolRegistry, scripted: LLMResponse
    ) -> None:
        client = FakeLLMClient([scripted])

        response, history = await run_tool_loop(client, [QUESTION], registry.to_anthropic_tools(), registry)

        # The loop returns these like a final answer (#84).
        assert response is scripted
        assert len(client.requests) == 1
        assert history == [QUESTION]

    @pytest.mark.asyncio
    async def test_extra_generate_call_fails_loudly(self, registry: ToolRegistry) -> None:
        client = FakeLLMClient([fake_tool_calls(fake_tool_call("calculator", {"expression": "2 + 2"}))])

        with pytest.raises(AssertionError, match="no scripted response left"):
            await run_tool_loop(client, [QUESTION], registry.to_anthropic_tools(), registry)
