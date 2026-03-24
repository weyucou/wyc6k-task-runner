"""Agent runner service for executing conversations with tool support."""

import logging
from typing import Any

from commons.rate_limiter import rate_limiter_registry
from memory.models import Message, MessageRole as MemoryMessageRole

from agents.llm import LLMMessage, LLMResponse
from agents.llm.base import ToolCall
from agents.llm.factory import create_client_from_agent
from agents.tools import ToolRegistry
from agents.tools.builtin import register_builtin_tools

logger = logging.getLogger(__name__)


def session_messages_to_llm_messages(records: list[Any]) -> list[LLMMessage]:
    """Convert a list of memory.Message records to LLMMessage objects.

    System messages are skipped — system prompt is handled separately.
    Assistant messages reconstruct tool_calls from raw_data if present.
    Tool result messages use tool_call_id and tool_name from the record.

    Args:
        records: Ordered list of memory.Message instances.

    Returns:
        List of LLMMessage objects in the same order.
    """
    messages: list[LLMMessage] = []
    for record in records:
        if record.role == MemoryMessageRole.USER:
            messages.append(LLMMessage.user(record.content))
        elif record.role == MemoryMessageRole.ASSISTANT:
            tool_calls = None
            raw_tool_calls = record.raw_data.get("tool_calls") if record.raw_data else None
            if raw_tool_calls:
                tool_calls = [
                    ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"]) for tc in raw_tool_calls
                ]
            messages.append(LLMMessage.assistant(record.content, tool_calls))
        elif record.role == MemoryMessageRole.TOOL:
            messages.append(
                LLMMessage.tool_result(
                    tool_call_id=record.tool_call_id,
                    content=record.content,
                    name=record.tool_name or None,
                )
            )
        # Skip system messages — handled via system_prompt parameter
    return messages


class AgentRunner:
    """Service for running agent conversations with tool support.

    The runner handles:
    - LLM client creation and management
    - Tool registration and execution
    - Rate limiting
    - Conversation state management
    """

    def __init__(
        self,
        agent: Any | None = None,
        *,
        registry: ToolRegistry | None = None,
        register_builtins: bool = True,
        session_id: int | None = None,
    ) -> None:
        """Initialize the agent runner.

        Args:
            agent: Optional Agent model instance.
            registry: Optional custom tool registry.
            register_builtins: Whether to register built-in tools.
            session_id: Optional session ID for memory search context.
        """
        self.agent = agent
        self.registry = registry or ToolRegistry()
        self._client = None

        if register_builtins:
            agent_id = agent.id if agent else None
            register_builtin_tools(self.registry, agent_id=agent_id, session_id=session_id)

    async def get_client(self) -> Any:
        """Get or create the LLM client."""
        if self._client is None:
            if self.agent:
                self._client = await create_client_from_agent(self.agent)
            else:
                raise ValueError("No agent configured")
        return self._client

    def register_tool(self, tool: Any) -> None:
        """Register a tool with the runner.

        Args:
            tool: BaseTool instance to register.
        """
        self.registry.register(tool)

    def get_tools_for_provider(self, provider: str) -> list[dict[str, Any]]:
        """Get tool definitions formatted for the specified provider.

        Args:
            provider: LLM provider name.

        Returns:
            List of tool definitions.
        """
        provider_lower = provider.lower()

        if provider_lower == "anthropic":
            return self.registry.to_anthropic_tools()
        if provider_lower in ("openai", "vllm"):
            return self.registry.to_openai_tools()
        if provider_lower == "gemini":
            return self.registry.to_gemini_tools()
        if provider_lower == "ollama":
            return self.registry.to_openai_tools()  # Ollama uses OpenAI format

        # Default to OpenAI format
        return self.registry.to_openai_tools()

    async def _check_rate_limit(self) -> None:
        """Check and apply rate limiting if enabled."""
        if not self.agent or not self.agent.rate_limit_enabled:
            return

        limiter = rate_limiter_registry.get_or_create(
            agent_id=self.agent.id,
            rpm=self.agent.rate_limit_rpm,
        )
        await limiter.acquire_async()

    async def _load_session_transcript(self, session: Any, n: int) -> list[LLMMessage]:
        """Load the last N messages from a session and convert to LLMMessage objects.

        Args:
            session: memory.Session instance.
            n: Maximum number of messages to load.

        Returns:
            List of LLMMessage objects in chronological order.
        """
        records: list[Any] = []
        async for record in Message.objects.filter(session=session).order_by("-created_datetime")[:n]:
            records.append(record)
        records.reverse()
        return session_messages_to_llm_messages(records)

    async def run(
        self,
        messages: list[LLMMessage],
        *,
        session: Any | None = None,
        system_prompt: str | None = None,
        enable_tools: bool = True,
        tool_names: list[str] | None = None,
        max_tool_iterations: int = 10,
    ) -> tuple[LLMResponse, list[LLMMessage]]:
        """Run a conversation with the agent.

        Args:
            messages: Conversation messages.
            session: Optional memory.Session instance for transcript replay.
            system_prompt: Optional system prompt override.
            enable_tools: Whether to enable tool calling.
            tool_names: Optional list of tool names to enable.
            max_tool_iterations: Maximum tool calling iterations.

        Returns:
            Tuple of (final response, updated message history).
        """
        # Apply rate limiting
        await self._check_rate_limit()

        # Inject prior session transcript when configured
        if session is not None and self.agent is not None and self.agent.context_window_messages > 0:
            transcript = await self._load_session_transcript(session, self.agent.context_window_messages)
            if transcript:
                messages = transcript + messages

        # Get client
        client = await self.get_client()

        # Get system prompt
        prompt = system_prompt
        if prompt is None and self.agent:
            prompt = self.agent.system_prompt

        # Get tools if enabled
        tools = None
        if enable_tools and self.registry.list_tools():
            provider = self.agent.provider if self.agent else "openai"
            tools = self.get_tools_for_provider(provider)

            # Filter to specific tools if requested
            if tool_names:
                tools = [
                    t for t in tools if t.get("name") in tool_names or t.get("function", {}).get("name") in tool_names
                ]

        # Get generation parameters
        temperature = self.agent.temperature if self.agent else 0.7
        max_tokens = self.agent.max_tokens if self.agent else 4096

        if tools:
            # Run with tool execution loop
            response, history = await client.generate_with_tools(
                messages,
                tools,
                self.registry,
                system_prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                max_iterations=max_tool_iterations,
            )
        else:
            # Simple generation without tools
            response = await client.generate(
                messages,
                system_prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            history = messages

        return response, history

    async def chat(
        self,
        user_message: str,
        *,
        conversation_history: list[LLMMessage] | None = None,
        system_prompt: str | None = None,
        enable_tools: bool = True,
    ) -> tuple[str, list[LLMMessage]]:
        """Simple chat interface.

        Args:
            user_message: The user's message.
            conversation_history: Optional existing conversation.
            system_prompt: Optional system prompt.
            enable_tools: Whether to enable tools.

        Returns:
            Tuple of (assistant response text, updated history).
        """
        messages = list(conversation_history or [])
        messages.append(LLMMessage.user(user_message))

        response, history = await self.run(
            messages,
            system_prompt=system_prompt,
            enable_tools=enable_tools,
        )

        # Add assistant response to history
        history.append(LLMMessage.assistant(response.content, response.tool_calls if response.has_tool_calls else None))

        return response.content, history


async def run_agent_message(
    agent: Any,
    user_message: str,
    *,
    conversation_history: list[dict[str, Any]] | None = None,
    enable_tools: bool = True,
) -> dict[str, Any]:
    """Convenience function to run a single message through an agent.

    Args:
        agent: Agent model instance.
        user_message: User's message.
        conversation_history: Optional previous messages as dicts.
        enable_tools: Whether to enable tool calling.

    Returns:
        Dict with response content and metadata.
    """
    runner = AgentRunner(agent)

    # Convert history from dicts if provided
    messages = []
    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                messages.append(LLMMessage.system(content))
            elif role == "assistant":
                messages.append(LLMMessage.assistant(content))
            else:
                messages.append(LLMMessage.user(content))

    response_text, history = await runner.chat(
        user_message,
        conversation_history=messages,
        enable_tools=enable_tools,
    )

    return {
        "content": response_text,
        "history": [msg.to_dict() for msg in history],
        "agent": agent.name,
        "model": agent.model_name,
    }
