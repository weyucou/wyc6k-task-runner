"""Ollama LLM client with tool support."""

import json
import logging
import os
from typing import Any

from marvin_manager.llm.base import (
    BaseLLMClient,
    LLMMessage,
    LLMResponse,
    MessageRole,
    StopReason,
    ToolCall,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama3.2"
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaClient(BaseLLMClient):
    """Ollama client with tool support.

    Ollama supports tool calling via its chat API for compatible models.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama client."""
        default_base_url = os.getenv("DEFAULT_OLLAMA_BASE_URL", _DEFAULT_BASE_URL)
        default_model = os.getenv("DEFAULT_OLLAMA_MODEL", _DEFAULT_MODEL)
        super().__init__(
            api_key,
            base_url or default_base_url,
            model or default_model,
            **kwargs,
        )
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create the Ollama client."""
        if self._client is None:
            try:
                from ollama import AsyncClient  # noqa: PLC0415
            except ImportError as err:
                raise ImportError("ollama package required: uv add ollama") from err

            self._client = AsyncClient(host=self.base_url)
        return self._client

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert messages to Ollama's format."""
        result = []

        for msg in messages:
            if msg.role == MessageRole.TOOL:
                # Tool results
                result.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                    }
                )
            elif msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                # Assistant with tool calls
                tool_calls_formatted = [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    }
                    for tc in msg.tool_calls
                ]
                result.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
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

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert tool definitions to Ollama format."""
        ollama_tools = []

        for tool in tools:
            # Handle different input formats
            if "function" in tool:
                # OpenAI format
                ollama_tools.append(tool)
            elif "input_schema" in tool:
                # Anthropic format
                ollama_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool["input_schema"],
                        },
                    }
                )
            else:
                # Direct format
                ollama_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        },
                    }
                )

        return ollama_tools

    def _parse_response(self, response: dict[str, Any]) -> LLMResponse:
        """Parse Ollama response into LLMResponse."""
        message = response.get("message", {})
        content = message.get("content", "")
        tool_calls = []

        # Parse tool calls
        if "tool_calls" in message:
            for i, tc in enumerate(message["tool_calls"]):
                func = tc.get("function", {})
                arguments = func.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}

                tool_calls.append(
                    ToolCall(
                        id=f"call_{i}",
                        name=func.get("name", "unknown"),
                        arguments=arguments,
                    )
                )

        # Determine stop reason
        stop_reason = StopReason.END_TURN
        if tool_calls:
            stop_reason = StopReason.TOOL_USE
        elif response.get("done_reason") == "length":
            stop_reason = StopReason.MAX_TOKENS

        # Token counts
        input_tokens = response.get("prompt_eval_count", 0)
        output_tokens = response.get("eval_count", 0)

        return LLMResponse(
            content=content,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=response.get("model", self.model or ""),
            raw_response=response,
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
        """Generate a response from Ollama."""
        client = self._get_client()

        # Prepare messages with system prompt
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(self._convert_messages(messages))

        # Build options
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }

        if stop_sequences:
            options["stop"] = stop_sequences

        # Build request
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "options": options,
        }

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        try:
            response = await client.chat(**request_kwargs)
            return self._parse_response(response)
        except ImportError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("Ollama API error")
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

            response = await self.generate(
                conversation,
                system_prompt=system_prompt,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if not response.has_tool_calls:
                return response, conversation

            conversation.append(LLMMessage.assistant(response.content, response.tool_calls))

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

                conversation.append(
                    LLMMessage.tool_result(
                        tool_call_id=tool_call.id,
                        content=tool_output,
                        name=tool_call.name,
                    )
                )

        logger.warning("Max tool iterations reached")
        final_response = await self.generate(
            conversation,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return final_response, conversation
