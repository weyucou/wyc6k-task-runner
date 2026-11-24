# Copyright (c) 2026 MNKT
"""Tests for LLM client request kwargs, checked against the installed provider SDKs."""

import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic import AsyncAnthropic
from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI

from marvin.llm.anthropic_client import AnthropicClient
from marvin.llm.base import LLMMessage, StopReason, ToolCall
from marvin.llm.gemini_client import GeminiClient
from marvin.llm.openai_client import OpenAIClient
from marvin.loop import run_tool_loop
from marvin.tools import ToolRegistry
from marvin.tools.builtin import CalculatorTool

SAMPLING_PARAMS = {"temperature", "top_p", "top_k"}
MESSAGES = [LLMMessage.user("What is 2 + 2?")]


def _assert_sdk_accepts(sdk_method: Any, kwargs: dict[str, Any]) -> None:
    """Raise TypeError when the installed SDK method has no parameter for a request kwarg."""
    inspect.signature(sdk_method).bind(**kwargs)


def _tool_results_turn() -> list[LLMMessage]:
    """A conversation whose last turn called two tools: the first succeeded, the second failed."""
    calls = [
        ToolCall(id="call_1", name="calculator", arguments={"expression": "2 + 2"}),
        ToolCall(id="call_2", name="calculator", arguments={"expression": "1 / 0"}),
    ]
    return [
        *MESSAGES,
        LLMMessage.assistant("", calls),
        LLMMessage.tool_result("call_1", "4", name="calculator"),
        LLMMessage.tool_result("call_2", "Error: division by zero", name="calculator", is_error=True),
    ]


class TestLLMMessage:
    def test_to_dict_flags_only_failed_tool_results(self) -> None:
        *_, ok, failed = _tool_results_turn()

        assert "is_error" not in ok.to_dict()
        assert failed.to_dict()["is_error"] is True


def _anthropic_message(block: SimpleNamespace, stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[block],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        model="claude-sonnet-5-5",
    )


def _anthropic_client(*replies: SimpleNamespace) -> tuple[AnthropicClient, AsyncMock]:
    create = AsyncMock(side_effect=list(replies))
    client = AnthropicClient(api_key="x")
    client._client = MagicMock()
    client._client.messages.create = create
    return client, create


ANTHROPIC_TEXT_REPLY = _anthropic_message(SimpleNamespace(type="text", text="4"), "end_turn")
ANTHROPIC_TOOL_REPLY = _anthropic_message(
    SimpleNamespace(type="tool_use", id="toolu_1", name="calculator", input={"expression": "2 + 2"}),
    "tool_use",
)


class TestAnthropicClient:
    @pytest.mark.asyncio
    async def test_request_omits_sampling_params(self) -> None:
        client, create = _anthropic_client(ANTHROPIC_TEXT_REPLY)
        tools = [CalculatorTool().to_anthropic_format()]

        response = await client.generate(MESSAGES, system_prompt="Be brief.", tools=tools, temperature=0.3)

        assert response.stop_reason == StopReason.END_TURN
        kwargs = create.call_args.kwargs
        assert not SAMPLING_PARAMS & kwargs.keys()
        _assert_sdk_accepts(AsyncAnthropic(api_key="x").messages.create, kwargs)

    @pytest.mark.asyncio
    async def test_tool_loop_omits_sampling_params(self) -> None:
        # One tool round, then the max-iterations final call: covers both generate() call sites.
        client, create = _anthropic_client(ANTHROPIC_TOOL_REPLY, ANTHROPIC_TEXT_REPLY)
        registry = ToolRegistry()
        registry.register(CalculatorTool())

        response, _ = await run_tool_loop(
            client,
            MESSAGES,
            registry.to_anthropic_tools(),
            registry,
            temperature=0.3,
            max_iterations=1,
        )

        assert response.stop_reason == StopReason.END_TURN
        assert create.await_count == 2
        for call in create.call_args_list:
            assert not SAMPLING_PARAMS & call.kwargs.keys()

    @pytest.mark.asyncio
    async def test_failed_tool_result_sets_is_error(self) -> None:
        client, create = _anthropic_client(ANTHROPIC_TEXT_REPLY)

        await client.generate(_tool_results_turn())

        *_, ok, failed = create.call_args.kwargs["messages"]
        assert ok["content"] == [{"type": "tool_result", "tool_use_id": "call_1", "content": "4"}]
        assert failed["content"] == [
            {"type": "tool_result", "tool_use_id": "call_2", "content": "Error: division by zero", "is_error": True}
        ]

    def test_default_model(self, monkeypatch) -> None:
        monkeypatch.delenv("DEFAULT_ANTHROPIC_MODEL", raising=False)
        assert AnthropicClient().model == "claude-sonnet-5-5"


def _gemini_client() -> tuple[GeminiClient, AsyncMock]:
    generate_content = AsyncMock(
        return_value=SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[SimpleNamespace(text="4", function_call=None)]),
                    finish_reason="STOP",
                )
            ],
            usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
        )
    )
    client = GeminiClient(api_key="x")
    client._client = MagicMock()
    client._client.aio.models.generate_content = generate_content
    return client, generate_content


class TestGeminiClient:
    @pytest.mark.asyncio
    async def test_tools_passed_in_config(self) -> None:
        client, generate_content = _gemini_client()
        tools = [CalculatorTool().to_gemini_format()]

        response = await client.generate(MESSAGES, system_prompt="Be brief.", tools=tools, temperature=0.3)

        assert response.stop_reason == StopReason.END_TURN
        kwargs = generate_content.call_args.kwargs
        assert "tools" not in kwargs
        assert kwargs["config"]["tools"][0]["function_declarations"][0]["name"] == "calculator"
        assert kwargs["config"]["temperature"] == 0.3
        _assert_sdk_accepts(genai.Client(api_key="x").aio.models.generate_content, kwargs)
        genai_types.GenerateContentConfig.model_validate(kwargs["config"])

    @pytest.mark.asyncio
    async def test_failed_tool_result_is_sent_as_an_error(self) -> None:
        client, generate_content = _gemini_client()

        await client.generate(_tool_results_turn())

        contents = generate_content.call_args.kwargs["contents"]
        *_, ok, failed = contents
        assert ok["parts"] == [{"function_response": {"name": "calculator", "response": {"result": "4"}}}]
        assert failed["parts"] == [
            {"function_response": {"name": "calculator", "response": {"error": "Error: division by zero"}}}
        ]
        for content in contents:
            genai_types.Content.model_validate(content)

    def test_default_model(self, monkeypatch) -> None:
        monkeypatch.delenv("DEFAULT_GEMINI_MODEL", raising=False)
        assert GeminiClient().model == "gemini-3.6-flash"


def _openai_client() -> tuple[OpenAIClient, AsyncMock]:
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="4", tool_calls=None), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            model="served-model",
        )
    )
    client = OpenAIClient(api_key="x", base_url="http://localhost:8000/v1", model="served-model")
    client._client = MagicMock()
    client._client.chat.completions.create = create
    return client, create


class TestOpenAIClient:
    @pytest.mark.asyncio
    async def test_request_kwargs(self) -> None:
        client, create = _openai_client()
        tools = [CalculatorTool().to_openai_format()]

        response = await client.generate(MESSAGES, tools=tools, temperature=0.3, max_tokens=256)

        assert response.stop_reason == StopReason.END_TURN
        kwargs = create.call_args.kwargs
        assert kwargs.keys() == {"model", "messages", "max_tokens", "temperature", "tools", "tool_choice"}
        assert kwargs["temperature"] == 0.3
        assert kwargs["tool_choice"] == "auto"
        _assert_sdk_accepts(AsyncOpenAI(api_key="x").chat.completions.create, kwargs)
