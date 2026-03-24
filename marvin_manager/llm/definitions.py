"""Definitions and types for LLM clients."""

from typing import Any

from pydantic import BaseModel, Field

from marvin_manager.definitions import StringEnumWithChoices


class AnthropicBlockTypes(StringEnumWithChoices):
    """Anthropic response block types."""

    TEXT = "text"
    TOOL_USE = "tool_use"


class AnthropicContentTypes(StringEnumWithChoices):
    """Anthropic message content types."""

    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class AnthropicRoles(StringEnumWithChoices):
    """Anthropic message roles."""

    USER = "user"
    ASSISTANT = "assistant"


class AnthropicStopReasons(StringEnumWithChoices):
    """Anthropic stop reasons."""

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"
    STOP_SEQUENCE = "stop_sequence"


class AnthropicToolUseBlock(BaseModel):
    """Anthropic tool use content block."""

    type: str = AnthropicContentTypes.TOOL_USE.value
    id: str
    name: str
    input: dict[str, Any]


class AnthropicToolResultBlock(BaseModel):
    """Anthropic tool result content block."""

    type: str = AnthropicContentTypes.TOOL_RESULT.value
    tool_use_id: str
    content: str


class AnthropicTextBlock(BaseModel):
    """Anthropic text content block."""

    type: str = AnthropicContentTypes.TEXT.value
    text: str


class AnthropicMessage(BaseModel):
    """Anthropic message format."""

    role: str
    content: str | list[dict[str, Any]]


class AnthropicRequest(BaseModel):
    """Anthropic API request parameters."""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    system: str | None = None
    tools: list[dict[str, Any]] | None = None
    stop_sequences: list[str] | None = None
