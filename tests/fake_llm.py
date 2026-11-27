# Copyright (c) 2026 MNKT
"""Scripted in-process LLM client for tool-loop tests.

`FakeLLMClient` returns queued `LLMResponse`s in order and records each request it
receives. Build the responses with `fake_text`, `fake_tool_calls`, `fake_error` and
`fake_max_tokens`. `RecordingTool` records each call it receives.
"""

import itertools
from dataclasses import dataclass
from typing import Any

from marvin.llm.base import BaseLLMClient, LLMMessage, LLMResponse, StopReason, ToolCall
from marvin.tools.base import BaseTool, ToolResult

_call_ids = itertools.count(1)


class RecordingTool(BaseTool):
    """A tool named `name` that records the arguments of each call in `calls`."""

    description = "Records each call."
    parameters = []

    def __init__(self, name: str, *, require_approval: bool = False) -> None:
        super().__init__()
        self.name = name
        self.require_approval = require_approval
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult.success(f"{self.name} ran")


@dataclass
class FakeRequest:
    """One `generate` call received by `FakeLLMClient`."""

    messages: list[LLMMessage]
    system_prompt: str | None
    tools: list[dict[str, Any]] | None
    temperature: float
    max_tokens: int


class FakeLLMClient(BaseLLMClient):
    """Returns scripted responses in order and records each request in `requests`."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(model="fake")
        self._responses = list(responses)
        self.requests: list[FakeRequest] = []

    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """Record the request and return the next scripted response."""
        self.requests.append(
            FakeRequest(
                messages=list(messages),
                system_prompt=system_prompt,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        if not self._responses:
            raise AssertionError("no scripted response left")
        return self._responses.pop(0)


def fake_text(content: str) -> LLMResponse:
    """A final answer: text with no tool calls."""
    return LLMResponse(content=content, stop_reason=StopReason.END_TURN)


def fake_tool_call(name: str, arguments: dict[str, Any], call_id: str | None = None) -> ToolCall:
    """A tool call. `call_id` defaults to a fresh `call_<n>`."""
    return ToolCall(id=call_id or f"call_{next(_call_ids)}", name=name, arguments=arguments)


def fake_tool_calls(*calls: ToolCall, content: str = "") -> LLMResponse:
    """A response that requests `calls`, in order."""
    return LLMResponse(content=content, stop_reason=StopReason.TOOL_USE, tool_calls=list(calls))


def fake_error(message: str) -> LLMResponse:
    """A failed API call, in the shape each provider client returns."""
    return LLMResponse(content=f"Error: {message}", stop_reason=StopReason.ERROR)


def fake_max_tokens(content: str) -> LLMResponse:
    """A response cut off at the `max_tokens` limit."""
    return LLMResponse(content=content, stop_reason=StopReason.MAX_TOKENS)
