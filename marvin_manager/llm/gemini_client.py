"""Google Gemini LLM client with tool support."""

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

_DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiClient(BaseLLMClient):
    """Google Gemini client with function calling support."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Gemini client."""
        default_model = os.getenv("DEFAULT_GEMINI_MODEL", _DEFAULT_MODEL)
        super().__init__(api_key, base_url, model or default_model, **kwargs)
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create the Gemini client."""
        if self._client is None:
            try:
                from google import genai  # noqa: PLC0415
            except ImportError as err:
                raise ImportError("google-genai package required: uv add google-genai") from err

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Convert messages to Gemini's format."""
        result = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # System messages handled separately
                continue

            if msg.role == MessageRole.TOOL:
                # Tool results use function_response
                result.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": msg.name or "unknown",
                                    "response": {"result": msg.content},
                                }
                            }
                        ],
                    }
                )
            elif msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                # Assistant with function calls
                parts: list[dict[str, Any]] = []
                if msg.content:
                    parts.append({"text": msg.content})
                for tc in msg.tool_calls:
                    parts.append(
                        {
                            "function_call": {
                                "name": tc.name,
                                "args": tc.arguments,
                            }
                        }
                    )
                result.append({"role": "model", "parts": parts})
            elif msg.role == MessageRole.ASSISTANT:
                result.append(
                    {
                        "role": "model",
                        "parts": [{"text": msg.content}],
                    }
                )
            else:
                result.append(
                    {
                        "role": "user",
                        "parts": [{"text": msg.content}],
                    }
                )

        return result

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert tool definitions to Gemini format."""
        gemini_functions = []

        for tool in tools:
            # Handle both Anthropic and OpenAI formats
            if "function" in tool:
                # OpenAI format
                func = tool["function"]
                gemini_functions.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    }
                )
            else:
                # Anthropic format or direct format
                gemini_functions.append(
                    {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", tool.get("parameters", {})),
                    }
                )

        return [{"function_declarations": gemini_functions}]

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Gemini response into LLMResponse."""
        content_parts = []
        tool_calls = []
        tool_call_counter = 0

        # Extract parts from response
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        content_parts.append(part.text)
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{tool_call_counter}",
                                name=fc.name,
                                arguments=dict(fc.args) if fc.args else {},
                            )
                        )
                        tool_call_counter += 1

        # Determine stop reason
        stop_reason = StopReason.END_TURN
        if tool_calls:
            stop_reason = StopReason.TOOL_USE
        elif hasattr(response, "candidates") and response.candidates:
            finish_reason = getattr(response.candidates[0], "finish_reason", None)
            if finish_reason and "MAX" in str(finish_reason):
                stop_reason = StopReason.MAX_TOKENS

        # Token usage
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata"):
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

        return LLMResponse(
            content="\n".join(content_parts),
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model or _DEFAULT_MODEL,
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
        """Generate a response from Gemini."""
        client = self._get_client()

        # Build config
        config: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        if system_prompt:
            config["system_instruction"] = system_prompt

        if stop_sequences:
            config["stop_sequences"] = stop_sequences

        # Convert messages
        contents = self._convert_messages(messages)

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "contents": contents,
            "config": config,
        }

        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        try:
            response = await client.aio.models.generate_content(**request_kwargs)
            return self._parse_response(response)
        except ImportError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("Gemini API error")
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

            # Add assistant message with tool calls
            conversation.append(LLMMessage.assistant(response.content, response.tool_calls))

            # Execute tools
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
