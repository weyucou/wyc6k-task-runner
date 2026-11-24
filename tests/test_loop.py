# Copyright (c) 2026 MNKT
"""Tests for the shared tool-call loop, driven by the scripted FakeLLMClient."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from marvin.llm.base import LLMMessage, MessageRole
from marvin.loop import (
    CONTINUE_PROMPT,
    MAX_CONTINUATIONS,
    REISSUE_TOOL_CALL,
    FailureReason,
    RunOutcome,
    run_tool_loop,
)
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
CONTINUE = LLMMessage.user(CONTINUE_PROMPT)


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


class RecordingTool(BaseTool):
    """Records the arguments of each call it runs."""

    name = "recording"
    description = "Records its calls."
    parameters = []

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult.success("recorded")


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

    result = await run_tool_loop(client, [QUESTION], registry.to_anthropic_tools(), registry)

    assert result.text == "done"
    assert len(client.requests) == 2
    tool_result = result.history[-1]
    assert (tool_result.role, tool_result.tool_call_id, tool_result.name) == (MessageRole.TOOL, call.id, name)
    return tool_result


class TestRunToolLoop:
    @pytest.mark.asyncio
    async def test_final_answer_without_tool_calls(self, registry: ToolRegistry) -> None:
        answer = fake_text("4")
        client = FakeLLMClient([answer])
        messages = [QUESTION]
        tools = registry.to_anthropic_tools()

        result = await run_tool_loop(client, messages, tools, registry)

        assert result.response is answer
        assert (result.outcome, result.reason, result.text) == (RunOutcome.SUCCEEDED, None, "4")
        assert result.succeeded
        assert len(client.requests) == 1
        assert client.requests[0].messages == [QUESTION]
        assert client.requests[0].tools == tools
        assert result.history == [QUESTION]
        assert result.history is not messages
        assert messages == [QUESTION]

    @pytest.mark.asyncio
    async def test_one_tool_round_trip(self, registry: ToolRegistry) -> None:
        call = fake_tool_call("calculator", {"expression": "2 + 2"})
        client = FakeLLMClient([fake_tool_calls(call, content="Let me check."), fake_text("2 + 2 = 4")])
        messages = [QUESTION]

        result = await run_tool_loop(client, messages, registry.to_anthropic_tools(), registry)

        assert (result.outcome, result.text) == (RunOutcome.SUCCEEDED, "2 + 2 = 4")
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
        assert result.history == [user, assistant, tool_result]
        assert messages == [QUESTION]

    @pytest.mark.asyncio
    async def test_several_tool_calls_in_one_turn(self, registry: ToolRegistry) -> None:
        first = fake_tool_call("calculator", {"expression": "2 + 2"})
        second = fake_tool_call("calculator", {"expression": "3 * 3"})
        client = FakeLLMClient([fake_tool_calls(first, second), fake_text("4 and 9")])

        result = await run_tool_loop(client, [QUESTION], registry.to_anthropic_tools(), registry)

        history = result.history
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

        result = await run_tool_loop(
            client,
            [QUESTION],
            tools,
            registry,
            system_prompt="Be brief.",
            temperature=0.2,
            max_tokens=256,
            max_iterations=2,
        )

        # The run did not finish: the final call's text is partial output.
        assert (result.outcome, result.reason, result.text) == (
            RunOutcome.FAILED,
            FailureReason.STEP_LIMIT,
            "Stopping here.",
        )
        assert [request.tools for request in client.requests] == [tools, tools, None]
        for request in client.requests:
            assert (request.system_prompt, request.temperature, request.max_tokens) == ("Be brief.", 0.2, 256)
        assert [m.content for m in result.history if m.role == MessageRole.TOOL] == ["2", "4"]
        assert client.requests[2].messages == result.history

    @pytest.mark.asyncio
    async def test_iteration_cap_keeps_the_text_of_a_truncated_turn(self, registry: ToolRegistry) -> None:
        client = FakeLLMClient([fake_max_tokens("The answer"), fake_text(" is 4.")])

        result = await run_tool_loop(client, [QUESTION], registry.to_anthropic_tools(), registry, max_iterations=1)

        assert (result.outcome, result.reason, result.text) == (
            RunOutcome.FAILED,
            FailureReason.STEP_LIMIT,
            "The answer is 4.",
        )
        assert client.requests[1].tools is None
        assert client.requests[1].messages == [QUESTION, LLMMessage.assistant("The answer"), CONTINUE]

    @pytest.mark.asyncio
    async def test_error_at_the_iteration_cap_is_an_llm_error(self, registry: ToolRegistry) -> None:
        tools = registry.to_anthropic_tools()
        client = FakeLLMClient(
            [fake_tool_calls(fake_tool_call("calculator", {"expression": "2 + 2"})), fake_error("overloaded")]
        )

        result = await run_tool_loop(client, [QUESTION], tools, registry, max_iterations=1)

        assert (result.outcome, result.reason, result.detail, result.text) == (
            RunOutcome.FAILED,
            FailureReason.LLM_ERROR,
            "Error: overloaded",
            "",
        )
        assert [request.tools for request in client.requests] == [tools, None]

    @pytest.mark.asyncio
    async def test_llm_error_fails_the_run(self, registry: ToolRegistry) -> None:
        error = fake_error("rate limited")
        client = FakeLLMClient([error])

        result = await run_tool_loop(client, [QUESTION], registry.to_anthropic_tools(), registry)

        assert result.response is error
        assert (result.outcome, result.reason, result.detail) == (
            RunOutcome.FAILED,
            FailureReason.LLM_ERROR,
            "Error: rate limited",
        )
        assert not result.succeeded
        # The error message is the failure's detail: it is neither output nor a turn in the history.
        assert result.text == ""
        assert result.history == [QUESTION]
        assert len(client.requests) == 1

    @pytest.mark.asyncio
    async def test_llm_error_after_a_tool_round(self, registry: ToolRegistry) -> None:
        call = fake_tool_call("calculator", {"expression": "2 + 2"})
        client = FakeLLMClient([fake_tool_calls(call), fake_error("overloaded")])

        result = await run_tool_loop(client, [QUESTION], registry.to_anthropic_tools(), registry)

        assert (result.reason, result.detail) == (FailureReason.LLM_ERROR, "Error: overloaded")
        assert len(client.requests) == 2
        # The tool ran before the error.
        assert [(m.role, m.content) for m in result.history] == [
            (MessageRole.USER, QUESTION.content),
            (MessageRole.ASSISTANT, ""),
            (MessageRole.TOOL, "4"),
        ]

    @pytest.mark.asyncio
    async def test_llm_error_keeps_the_earlier_partial_text(self, registry: ToolRegistry) -> None:
        client = FakeLLMClient([fake_max_tokens("The answer"), fake_error("overloaded")])

        result = await run_tool_loop(client, [QUESTION], registry.to_anthropic_tools(), registry)

        assert (result.reason, result.detail, result.text) == (
            FailureReason.LLM_ERROR,
            "Error: overloaded",
            "The answer",
        )

    @pytest.mark.asyncio
    async def test_truncated_tool_calls_are_not_run(self, registry: ToolRegistry) -> None:
        recorder = RecordingTool()
        registry.register(recorder)
        first = fake_tool_call("recording", {"note": "complete"})
        second = fake_tool_call("recording", {"raw": '{"note": "cut o'})
        client = FakeLLMClient([fake_max_tokens("Recording.", first, second), fake_text("Done.")])

        result = await run_tool_loop(client, [QUESTION], registry.to_anthropic_tools(), registry)

        assert recorder.calls == []
        assert (result.outcome, result.text) == (RunOutcome.SUCCEEDED, "Done.")
        _, assistant, *tool_results = client.requests[1].messages
        assert (assistant.role, assistant.content, assistant.tool_calls) == (
            MessageRole.ASSISTANT,
            "Recording.",
            [first, second],
        )
        assert [(m.role, m.tool_call_id, m.name, m.content, m.is_error) for m in tool_results] == [
            (MessageRole.TOOL, first.id, "recording", REISSUE_TOOL_CALL, True),
            (MessageRole.TOOL, second.id, "recording", REISSUE_TOOL_CALL, True),
        ]

    @pytest.mark.asyncio
    async def test_truncated_text_is_continued(self, registry: ToolRegistry) -> None:
        client = FakeLLMClient([fake_max_tokens("The answer"), fake_max_tokens(" is"), fake_text(" 4.")])

        result = await run_tool_loop(client, [QUESTION], registry.to_anthropic_tools(), registry)

        assert (result.outcome, result.reason, result.text) == (RunOutcome.SUCCEEDED, None, "The answer is 4.")
        assert client.requests[2].messages == [
            QUESTION,
            LLMMessage.assistant("The answer"),
            CONTINUE,
            LLMMessage.assistant(" is"),
            CONTINUE,
        ]
        assert result.history == client.requests[2].messages

    @pytest.mark.asyncio
    async def test_too_many_truncations_in_a_row_fail_with_output_limit(self, registry: ToolRegistry) -> None:
        # A truncated tool-call turn counts like a truncated text turn.
        truncated_call = fake_max_tokens("", fake_tool_call("calculator", {"raw": '{"expression": "2 +'}))
        parts = [f"part {n}. " for n in range(MAX_CONTINUATIONS)]
        client = FakeLLMClient([truncated_call, *(fake_max_tokens(part) for part in parts)])

        result = await run_tool_loop(client, [QUESTION], registry.to_anthropic_tools(), registry)

        assert (result.outcome, result.reason) == (RunOutcome.FAILED, FailureReason.OUTPUT_LIMIT)
        assert len(client.requests) == MAX_CONTINUATIONS + 1
        assert result.text == "".join(parts)
        # As in every run, the last turn is not in the history.
        assert result.history == client.requests[-1].messages

    @pytest.mark.asyncio
    async def test_a_complete_turn_resets_the_truncation_count(self, registry: ToolRegistry) -> None:
        call = fake_tool_call("calculator", {"expression": "2 + 2"})
        parts = [f"part {n}. " for n in range(MAX_CONTINUATIONS)]
        client = FakeLLMClient(
            [
                fake_max_tokens("Let me "),
                fake_tool_calls(call),
                *(fake_max_tokens(part) for part in parts),
                fake_text("Done."),
            ]
        )

        result = await run_tool_loop(client, [QUESTION], registry.to_anthropic_tools(), registry)

        assert (result.outcome, result.reason) == (RunOutcome.SUCCEEDED, None)
        assert len(client.requests) == MAX_CONTINUATIONS + 3
        # Text from before the tool turn is not part of the answer.
        assert result.text == "".join(parts) + "Done."

    @pytest.mark.asyncio
    async def test_extra_generate_call_fails_loudly(self, registry: ToolRegistry) -> None:
        client = FakeLLMClient([fake_tool_calls(fake_tool_call("calculator", {"expression": "2 + 2"}))])

        with pytest.raises(AssertionError, match="no scripted response left"):
            await run_tool_loop(client, [QUESTION], registry.to_anthropic_tools(), registry)
