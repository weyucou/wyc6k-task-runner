# Copyright (c) 2026 MNKT
"""Anthropic Claude LLM client with tool support."""

import logging
import os
from typing import Any

from marvin.llm.base import (
    BaseLLMClient,
    LLMMessage,
    LLMResponse,
    MessageRole,
    StopReason,
    ToolCall,
)
from marvin.llm.definitions import (
    AnthropicBlockTypes,
    AnthropicContentTypes,
    AnthropicRequest,
    AnthropicRoles,
    AnthropicStopReasons,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-5-5"


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client with native tool support."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Anthropic client."""
        default_model = os.getenv("DEFAULT_ANTHROPIC_MODEL", _DEFAULT_MODEL)
        super().__init__(api_key, base_url, model or default_model, **kwargs)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic  # noqa: PLC0415
            except ImportError as err:
                raise ImportError("anthropic package required: uv add anthropic") from err

            client_kwargs: dict[str, Any] = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self._client = AsyncAnthropic(**client_kwargs)
        return self._client

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert messages to Anthropic's format."""
        result = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # System messages are handled separately in Anthropic
                continue

            if msg.role == MessageRole.TOOL:
                # Tool results use tool_result content block
                result.append(
                    {
                        "role": AnthropicRoles.USER.value,
                        "content": [
                            {
                                "type": AnthropicContentTypes.TOOL_RESULT.value,
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )
            elif msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                # Assistant message with tool calls
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": AnthropicContentTypes.TEXT.value, "text": msg.content})
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": AnthropicContentTypes.TOOL_USE.value,
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                result.append({"role": AnthropicRoles.ASSISTANT.value, "content": content})
            else:
                result.append(
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                    }
                )

        return result

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Anthropic response into LLMResponse."""
        content_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == AnthropicBlockTypes.TEXT.value:
                content_parts.append(block.text)
            elif block.type == AnthropicBlockTypes.TOOL_USE.value:
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        # Map stop reason
        stop_reason_map = {
            AnthropicStopReasons.END_TURN.value: StopReason.END_TURN,
            AnthropicStopReasons.MAX_TOKENS.value: StopReason.MAX_TOKENS,
            AnthropicStopReasons.TOOL_USE.value: StopReason.TOOL_USE,
            AnthropicStopReasons.STOP_SEQUENCE.value: StopReason.STOP_SEQUENCE,
        }
        stop_reason = stop_reason_map.get(response.stop_reason, StopReason.END_TURN)

        return LLMResponse(
            content="\n".join(content_parts),
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )

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
        """Generate a response from Claude.

        `temperature` is ignored: API version 2026-10-21 rejects sampling parameters.
        """
        client = self._get_client()

        # Build request using pydantic model for validation
        request = AnthropicRequest(
            model=self.model or _DEFAULT_MODEL,
            messages=self._convert_messages(messages),  # type: ignore[arg-type]
            max_tokens=max_tokens,
            system=system_prompt,
            tools=tools,
            stop_sequences=stop_sequences,
        )

        # Convert to dict, excluding None values
        request_kwargs = request.model_dump(exclude_none=True)

        try:
            response = await client.messages.create(**request_kwargs)
            return self._parse_response(response)
        except ImportError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("Anthropic API error")
            return LLMResponse(
                content=f"Error: {e}",
                stop_reason=StopReason.ERROR,
                model=self.model or "",
            )
