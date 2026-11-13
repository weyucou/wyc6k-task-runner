# Copyright (c) 2026 MNKT
"""The tool-call loop shared by every LLM provider client."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from marvin.llm.base import BaseLLMClient, LLMMessage, LLMResponse, ToolCall
from marvin.tools import ToolRegistry

logger = logging.getLogger(__name__)

# Runs before each tool call. A returned string rejects the call: the tool does not run,
# and the model receives `Error: <reason>` as the tool result.
type BeforeToolCall = Callable[[ToolCall], Awaitable[str | None]]


async def run_tool_loop(
    client: BaseLLMClient,
    messages: list[LLMMessage],
    tools: list[dict[str, Any]],
    registry: ToolRegistry,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_iterations: int = 10,
    before_tool_call: BeforeToolCall | None = None,
) -> tuple[LLMResponse, list[LLMMessage]]:
    """Generate a response with automatic tool execution.

    1. Send the conversation to the LLM.
    2. If the LLM requests tool calls, pass each to `before_tool_call`, then execute the
       calls it does not reject, in order, through `registry`.
    3. Append the tool results and repeat until a response has no tool calls.

    After `max_iterations` rounds of tool calls, one final call is made without tools.

    Args:
        client: Provider client; called once per turn through `generate`.
        messages: Conversation messages. The list is copied, never mutated.
        tools: Tool definitions in the provider's format.
        registry: Registry that executes the tool calls.
        system_prompt: Optional system prompt.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        max_iterations: Maximum tool calling iterations.
        before_tool_call: Optional hook run before each tool call. A returned string rejects the
            call. If the hook raises, the call is not executed either.

    Returns:
        Tuple of (final response, message history). The final response is not appended to the history.
    """
    conversation = list(messages)
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        logger.debug("Tool iteration %d/%d", iteration, max_iterations)

        # Generate response
        response = await client.generate(
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
            try:
                reason = await before_tool_call(tool_call) if before_tool_call else None
                if reason is None:
                    logger.info("Executing tool: %s", tool_call.name)
                    result = await registry.execute(tool_call.name, tool_call.arguments)
                    tool_output = result.output
                else:
                    logger.warning("Rejected tool call %s: %s", tool_call.name, reason)
                    tool_output = f"Error: {reason}"
            except Exception as e:
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
    final_response = await client.generate(
        conversation,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return final_response, conversation
