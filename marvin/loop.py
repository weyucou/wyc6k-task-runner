# Copyright (c) 2026 MNKT
"""The tool-call loop shared by every LLM provider client."""

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from marvin.llm.base import BaseLLMClient, LLMMessage, LLMResponse, StopReason
from marvin.tools import ToolRegistry

logger = logging.getLogger(__name__)

# max_tokens stops in a row that the loop answers with a continuation or a reissue request.
# One more fails the run with output_limit.
MAX_CONTINUATIONS = 3
CONTINUE_PROMPT = (
    "Your response was cut off at the output token limit. "
    "Continue exactly where it stopped, without repeating any of it."
)
REISSUE_TOOL_CALL = (
    "Not run: the response hit the output token limit before this tool call was complete. "
    "Re-issue the tool call with complete arguments."
)


class RunOutcome(StrEnum):
    """How a run ended."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FailureReason(StrEnum):
    """Why a run failed."""

    LLM_ERROR = "llm_error"  # the provider client returned an error response
    STEP_LIMIT = "step_limit"  # max_iterations reached before a final answer
    OUTPUT_LIMIT = "output_limit"  # more than MAX_CONTINUATIONS max_tokens stops in a row


@dataclass
class LoopResult:
    """The end of a run.

    `response` is the last turn as the client returned it. `text` is the run's output: the
    text of the truncated turns since the last tool round, then the last turn's content.
    An error response adds nothing to `text`; its message is `detail`. `history` is the
    conversation up to, not including, the last turn.
    """

    response: LLMResponse
    text: str
    history: list[LLMMessage]
    outcome: RunOutcome
    reason: FailureReason | None = None
    detail: str = ""

    @property
    def succeeded(self) -> bool:
        """True when the run ended with a final answer."""
        return self.outcome == RunOutcome.SUCCEEDED


async def run_tool_loop(
    client: BaseLLMClient,
    messages: list[LLMMessage],
    tools: list[dict[str, Any]] | None,
    registry: ToolRegistry,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    max_iterations: int = 10,
) -> LoopResult:
    """Generate a response with automatic tool execution.

    1. Send the conversation to the LLM.
    2. If the LLM requests tool calls, execute them in order through `registry`.
    3. Append the tool results and repeat until a response has no tool calls.

    An error response fails the run (`llm_error`). A response cut off at `max_tokens` is not
    final: its tool calls are not run, and the model is asked to reissue them or to continue
    its text. More than `MAX_CONTINUATIONS` such responses in a row fail the run (`output_limit`).
    Every call counts toward `max_iterations`. After that many, one final call is made without
    tools, and the run fails (`step_limit`).

    Args:
        client: Provider client; called once per turn through `generate`.
        messages: Conversation messages. The list is copied, never mutated.
        tools: Tool definitions in the provider's format; None for a run without tools.
        registry: Registry that executes the tool calls.
        system_prompt: Optional system prompt.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        max_iterations: Maximum calls before the final call without tools.

    Returns:
        The run's outcome and output. The last response is not appended to the history.
    """
    conversation = list(messages)
    parts: list[str] = []  # the text of the truncated turns since the last tool round
    truncations = 0
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
        if response.stop_reason == StopReason.ERROR:
            return _llm_error(response, parts, conversation)

        if response.stop_reason == StopReason.MAX_TOKENS:
            truncations += 1
            logger.warning("Response cut off at max_tokens (%d in a row)", truncations)
            if truncations > MAX_CONTINUATIONS:
                text = "".join(parts) + response.content
                return LoopResult(response, text, conversation, RunOutcome.FAILED, FailureReason.OUTPUT_LIMIT)

            # Not final. Cut-off tool calls are not run: the model reissues them, or continues its text.
            conversation.append(LLMMessage.assistant(response.content, response.tool_calls or None))
            if response.has_tool_calls:
                conversation.extend(
                    LLMMessage.tool_result(call.id, REISSUE_TOOL_CALL, call.name, is_error=True)
                    for call in response.tool_calls
                )
            else:
                parts.append(response.content)
                conversation.append(LLMMessage.user(CONTINUE_PROMPT))
            continue

        # If no tool calls, we're done
        if not response.has_tool_calls:
            return LoopResult(response, "".join(parts) + response.content, conversation, RunOutcome.SUCCEEDED)

        truncations = 0
        parts.clear()

        # Add assistant message with tool calls
        conversation.append(LLMMessage.assistant(response.content, response.tool_calls))

        # Execute each tool and collect results
        for tool_call in response.tool_calls:
            logger.info("Executing tool: %s", tool_call.name)

            try:
                result = await registry.execute(tool_call.name, tool_call.arguments)
                tool_output = result.to_model_content()
                is_error = result.is_error
            except Exception as e:
                logger.exception("Tool execution failed: %s", tool_call.name)
                tool_output = f"Error executing tool: {e}"
                is_error = True

            # Add tool result to conversation
            conversation.append(
                LLMMessage.tool_result(
                    tool_call_id=tool_call.id,
                    content=tool_output,
                    name=tool_call.name,
                    is_error=is_error,
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
    if final_response.stop_reason == StopReason.ERROR:
        return _llm_error(final_response, parts, conversation)
    text = "".join(parts) + final_response.content
    return LoopResult(final_response, text, conversation, RunOutcome.FAILED, FailureReason.STEP_LIMIT)


def _llm_error(response: LLMResponse, parts: list[str], history: list[LLMMessage]) -> LoopResult:
    """Fail the run on an error response. Its message is the failure's detail, not output."""
    return LoopResult(response, "".join(parts), history, RunOutcome.FAILED, FailureReason.LLM_ERROR, response.content)
