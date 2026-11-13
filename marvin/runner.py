# Copyright (c) 2026 MNKT
"""Agent runner service for executing conversations with tool support."""

import logging
from typing import Any

from marvin.context import CustomerContextBundle
from marvin.llm import LLMMessage, LLMResponse
from marvin.llm.factory import create_client_from_agent_config
from marvin.loop import BeforeToolCall, run_tool_loop
from marvin.models import AgentConfig, ToolProfile
from marvin.rate_limiter import rate_limiter_registry
from marvin.tools import ToolRegistry
from marvin.tools.builtin import register_builtin_tools
from marvin.tools.policy import ToolPolicy

logger = logging.getLogger(__name__)


def _build_context_prefix(bundle: CustomerContextBundle) -> str:
    """Build a markdown context block from the fields of a CustomerContextBundle.

    Includes claude_md, sops (each file as a subsection), and project_goals.
    """
    parts = []
    if bundle.claude_md:
        parts.append(f"# CLAUDE.md\n\n{bundle.claude_md}")
    if bundle.sops:
        sops_sections = "\n\n".join(f"## {filename}\n\n{content}" for filename, content in bundle.sops.items())
        parts.append(f"# Standard Operating Procedures\n\n{sops_sections}")
    if bundle.project_goals:
        parts.append(f"# Project Goals\n\n{bundle.project_goals}")
    return "\n\n".join(parts)


class AgentRunner:
    """Service for running agent conversations with tool support.

    The runner handles:
    - LLM client creation and management
    - Tool registration and execution
    - Tool policy: the agent's allowed tools are the only ones sent to the model or run
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
        before_tool_call: BeforeToolCall | None = None,
    ) -> None:
        """Initialize the agent runner.

        Args:
            agent: Optional AgentConfig instance.
            registry: Optional custom tool registry.
            register_builtins: Whether to register built-in tools.
            session_id: Optional session ID for memory search context.
            context_bundle: Optional customer context bundle; when set, its
                claude_md, sops, and project_goals are prepended to every
                system prompt passed to run().
            before_tool_call: Optional hook run before each allowed tool call;
                a returned string rejects the call (see `ToolPolicy`).
        """
        self.agent = agent
        self.registry = registry or ToolRegistry()
        self._client = None
        self.context_bundle = context_bundle
        self.before_tool_call = before_tool_call

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

    def get_tools_for_provider(self, provider: str, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Get tool definitions formatted for the specified provider.

        Args:
            provider: LLM provider name.
            tool_names: Optional names of the tools to include; all registered tools when None.

        Returns:
            List of tool definitions.
        """
        provider_lower = provider.lower()

        if provider_lower == "anthropic":
            return self.registry.to_anthropic_tools(tool_names)
        if provider_lower in ("openai", "vllm"):
            return self.registry.to_openai_tools(tool_names)
        if provider_lower == "gemini":
            return self.registry.to_gemini_tools(tool_names)
        if provider_lower == "ollama":
            return self.registry.to_openai_tools(tool_names)  # Ollama uses OpenAI format

        # Default to OpenAI format
        return self.registry.to_openai_tools(tool_names)

    def _allowed_tools(self, tool_names: list[str] | None) -> list[str]:
        """Names of the tools a run may use: the agent's profile plus `tools_allow`, minus
        `tools_deny`, narrowed to `tool_names` when given. Without an agent, every registered tool.
        """
        registered = self.registry.list_tools()
        allowed = self.agent.get_allowed_tools(registered) if self.agent else registered
        if tool_names:
            allowed = [name for name in allowed if name in tool_names]
        return allowed

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

        Only the agent's allowed tools are sent to the model, and `ToolPolicy` rejects calls to
        any other tool. With no allowed tools, the model is called without tools.

        Args:
            messages: Conversation messages.
            system_prompt: Optional system prompt override.
            enable_tools: Whether to enable tool calling.
            tool_names: Optional tool names that narrow the agent's allowed tools.
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

        allowed = self._allowed_tools(tool_names) if enable_tools else []
        temperature = self.agent.temperature if self.agent else 0.7
        max_tokens = self.agent.max_tokens if self.agent else 4096

        if allowed:
            provider = self.agent.provider if self.agent else "openai"
            profile = self.agent.tool_profile if self.agent else ToolProfile.FULL
            policy = ToolPolicy(self.registry, allowed, profile=profile, before_tool_call=self.before_tool_call)
            response, history = await run_tool_loop(
                client,
                messages,
                self.get_tools_for_provider(provider, allowed),
                self.registry,
                system_prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                max_iterations=max_tool_iterations,
                before_tool_call=policy,
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
