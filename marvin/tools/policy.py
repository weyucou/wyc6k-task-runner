# Copyright (c) 2026 MNKT
"""`ToolPolicy`: the `before_tool_call` hook that `AgentRunner` passes to `run_tool_loop`."""

import logging
from typing import TYPE_CHECKING

from marvin.llm.base import ToolCall
from marvin.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from marvin.loop import BeforeToolCall

logger = logging.getLogger(__name__)


class ToolPolicy:
    """Rejects calls to tools outside a run's allowed set, then defers to the caller's hook.

    Every run is unattended, so nobody can approve a `require_approval` tool. An allowed one
    runs, and each call is logged. To keep an agent from using a tool, put it in `tools_deny`.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        allowed: list[str],
        *,
        profile: str,
        before_tool_call: BeforeToolCall | None = None,
    ) -> None:
        """Initialize the policy.

        Args:
            registry: Registry that holds the tools.
            allowed: Names of the tools the run may call.
            profile: The agent's tool profile, named in rejection messages.
            before_tool_call: Optional caller hook, run for allowed calls only.
        """
        self.registry = registry
        self.allowed = frozenset(allowed)
        self.profile = profile
        self.before_tool_call = before_tool_call

    async def __call__(self, tool_call: ToolCall) -> str | None:
        """Return the reason to reject `tool_call`, or None to let it run."""
        name = tool_call.name
        if name not in self.allowed:
            available = ", ".join(sorted(self.allowed))
            return (
                f"Tool '{name}' is not available to this agent (profile: {self.profile}). Available tools: {available}."
            )

        if self.before_tool_call is not None:
            reason = await self.before_tool_call(tool_call)
            if reason is not None:
                return reason

        tool = self.registry.get(name)
        if tool is not None and tool.require_approval:
            logger.info("auto-approved %s (unattended run)", name)
        return None
