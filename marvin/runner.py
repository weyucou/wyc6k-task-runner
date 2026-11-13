# Copyright (c) 2026 MNKT
"""Agent runner service for executing conversations with tool support."""

import logging
from typing import Any

from marvin.context import CustomerContextBundle
from marvin.llm import LLMMessage, LLMResponse
from marvin.llm.factory import create_client_from_agent_config
from marvin.loop import run_tool_loop
from marvin.models import AgentConfig
from marvin.rate_limiter import rate_limiter_registry
from marvin.settings import MEMORY_CONTEXT_MAX_CHARS
from marvin.tools import ToolRegistry
from marvin.tools.builtin import register_builtin_tools

logger = logging.getLogger(__name__)


def _build_memory_sections(bundle: CustomerContextBundle) -> list[str]:
    """Format MEMORY.md and the recent memory entries within MEMORY_CONTEXT_MAX_CHARS characters of memory text.

    MEMORY.md comes first, truncated to the budget. Entries follow newest first; the first entry
    that does not fit in the rest of the budget ends the list.
    """
    budget = MEMORY_CONTEXT_MAX_CHARS
    sections = []
    memory_index = bundle.memory_index[:budget]
    if memory_index:
        budget -= len(memory_index)
        sections.append(f"# Memory Index\n\n{memory_index}")
    entries = []
    for entry in sorted(bundle.daily_memories, key=lambda item: (item.date, item.filename), reverse=True):
        if len(entry.content) > budget:
            break
        budget -= len(entry.content)
        entries.append(f"## {entry.filename}\n\n{entry.content}")
    if entries:
        sections.append("# Recent Memory\n\n" + "\n\n".join(entries))
    return sections


def _build_context_prefix(bundle: CustomerContextBundle) -> str:
    """Build a markdown context block from the fields of a CustomerContextBundle.

    Includes claude_md, sops (each file as a subsection), project_goals, and the
    memory index and recent memory entries (bounded by MEMORY_CONTEXT_MAX_CHARS).
    """
    parts = []
    if bundle.claude_md:
        parts.append(f"# CLAUDE.md\n\n{bundle.claude_md}")
    if bundle.sops:
        sops_sections = "\n\n".join(f"## {filename}\n\n{content}" for filename, content in bundle.sops.items())
        parts.append(f"# Standard Operating Procedures\n\n{sops_sections}")
    if bundle.project_goals:
        parts.append(f"# Project Goals\n\n{bundle.project_goals}")
    parts.extend(_build_memory_sections(bundle))
    return "\n\n".join(parts)


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
        agent: AgentConfig | None = None,
        *,
        registry: ToolRegistry | None = None,
        register_builtins: bool = True,
        session_id: str | None = None,
        context_bundle: CustomerContextBundle | None = None,
    ) -> None:
        """Initialize the agent runner.

        Args:
            agent: Optional AgentConfig instance.
            registry: Optional custom tool registry.
            register_builtins: Whether to register built-in tools.
            session_id: Optional session ID for memory search context.
            context_bundle: Optional customer context bundle; when set, its
                claude_md, sops, project_goals and memory are prepended to
                every system prompt passed to run().
        """
        self.agent = agent
        self.registry = registry or ToolRegistry()
        self._client = None
        self.context_bundle = context_bundle

        if register_builtins:
            register_builtin_tools(self.registry, session_id=session_id)

    def get_client(self) -> Any:
        """Get or create the LLM client."""
        if self._client is None:
            if self.agent:
                self._client = create_client_from_agent_config(self.agent)
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
            agent_id=self.agent.name,
            rpm=self.agent.rate_limit_rpm,
        )
        await limiter.acquire_async()

    async def run(
        self,
        messages: list[LLMMessage],
        *,
        system_prompt: str | None = None,
        enable_tools: bool = True,
        tool_names: list[str] | None = None,
        max_tool_iterations: int = 10,
    ) -> tuple[LLMResponse, list[LLMMessage]]:
        """Run a conversation with the agent.

        Args:
            messages: Conversation messages.
            system_prompt: Optional system prompt override.
            enable_tools: Whether to enable tool calling.
            tool_names: Optional list of tool names to enable.
            max_tool_iterations: Maximum tool calling iterations.

        Returns:
            Tuple of (final response, updated message history).
        """
        await self._check_rate_limit()

        client = self.get_client()

        prompt = system_prompt
        if prompt is None and self.agent:
            prompt = self.agent.system_prompt or None

        if self.context_bundle is not None:
            prefix = _build_context_prefix(self.context_bundle)
            if prefix:
                prompt = f"{prefix}\n\n{prompt}" if prompt else prefix

        tools = None
        if enable_tools and self.registry.list_tools():
            provider = self.agent.provider if self.agent else "openai"
            tools = self.get_tools_for_provider(provider)

            if tool_names:
                tools = [
                    t for t in tools if t.get("name") in tool_names or t.get("function", {}).get("name") in tool_names
                ]

        temperature = self.agent.temperature if self.agent else 0.7
        max_tokens = self.agent.max_tokens if self.agent else 4096

        if tools:
            response, history = await run_tool_loop(
                client,
                messages,
                tools,
                self.registry,
                system_prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                max_iterations=max_tool_iterations,
            )
        else:
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

        history.append(LLMMessage.assistant(response.content, response.tool_calls if response.has_tool_calls else None))

        return response.content, history
