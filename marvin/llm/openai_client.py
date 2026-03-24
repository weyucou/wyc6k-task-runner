"""OpenAI-compatible LLM client with tool support.

This client works with:
- OpenAI API
- vLLM (OpenAI-compatible mode)
- Any OpenAI-compatible API
"""

import json
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

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o"


class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible client with function calling support."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI client."""
        default_model = os.getenv("DEFAULT_OPENAI_MODEL", _DEFAULT_MODEL)
        super().__init__(api_key, base_url, model or default_model, **kwargs)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI  # noqa: PLC0415
            except ImportError as err:
                raise ImportError("openai package required: uv add openai") from err

            client_kwargs: dict[str, Any] = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self._client = AsyncOpenAI(**client_kwargs)
        return self._client

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert messages to OpenAI's format."""
        result = []

        for msg in messages:
            if msg.role == MessageRole.TOOL:
                # Tool results use function role in OpenAI
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
            elif msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                # Assistant message with tool calls
                tool_calls_formatted = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                result.append(
                    {
                        "role": "assistant",
                        "content": msg.content or None,
                        "tool_calls": tool_calls_formatted,
                    }
                )
            else:
                result.append(
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                    }
                )

        return result

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse OpenAI response into LLMResponse."""
        choice = response.choices[0]
        message = choice.message

        content = message.content or ""
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": tc.function.arguments}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        # Map finish reason
        finish_reason_map = {
            "stop": StopReason.END_TURN,
            "length": StopReason.MAX_TOKENS,
            "tool_calls": StopReason.TOOL_USE,
            "function_call": StopReason.TOOL_USE,  # Legacy
        }
        stop_reason = finish_reason_map.get(choice.finish_reason or "stop", StopReason.END_TURN)

        # Token usage
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        return LLMResponse(
            content=content,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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
        """Generate a response from OpenAI-compatible API."""
        client = self._get_client()

        # Prepare messages with system prompt
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(self._convert_messages(messages))

        # Build request
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"

        if stop_sequences:
            request_kwargs["stop"] = stop_sequences

        try:
            response = await client.chat.completions.create(**request_kwargs)
            return self._parse_response(response)
        except ImportError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("OpenAI API error")
            return LLMResponse(
                content=f"Error: {e}",
                stop_reason=StopReason.ERROR,
                model=self.model or "",
            )

    async def generate_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        tool_executor: Any,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_iterations: int = 10,
    ) -> tuple[LLMResponse, list[LLMMessage]]:
        """Generate with automatic tool execution loop."""
        conversation = list(messages)
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.debug("Tool iteration %d/%d", iteration, max_iterations)

            # Generate response
            response = await self.generate(
                conversation,
                system_prompt=system_prompt,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # If no tool calls, we're done
            if not response.has_tool_calls:
                return response, conversation

            # Add assistant message with tool calls
            conversation.append(LLMMessage.assistant(response.content, response.tool_calls))

            # Execute each tool and collect results
            for tool_call in response.tool_calls:
                logger.info("Executing tool: %s", tool_call.name)

                try:
                    if hasattr(tool_executor, "execute"):
                        result = await tool_executor.execute(tool_call.name, tool_call.arguments)
                        tool_output = result.output if hasattr(result, "output") else str(result)
                    else:
                        result = await tool_executor(tool_call.name, tool_call.arguments)
                        tool_output = str(result)
                except Exception as e:  # noqa: BLE001
                    logger.exception("Tool execution failed: %s", tool_call.name)
                    tool_output = f"Error executing tool: {e}"

                # Add tool result to conversation
                conversation.append(
                    LLMMessage.tool_result(
                        tool_call_id=tool_call.id,
                        content=tool_output,
                        name=tool_call.name,
                    )
                )

        # Max iterations reached
        logger.warning("Max tool iterations reached")
        final_response = await self.generate(
            conversation,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return final_response, conversation
