"""Tool registry for managing available tools."""

import logging
from typing import Any

from marvin_manager.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing and executing tools.

    The registry stores tool instances and provides methods for:
    - Registering tools
    - Looking up tools by name
    - Executing tools with parameter validation
    - Converting tool definitions to LLM-specific formats
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance.

        Args:
            tool: The tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Args:
            name: The name of the tool to remove.
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug("Unregistered tool: %s", name)

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name.

        Args:
            name: The tool name.

        Returns:
            The tool instance or None if not found.
        """
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names.
        """
        return list(self._tools.keys())

    def get_all(self) -> list[BaseTool]:
        """Get all registered tools.

        Returns:
            List of tool instances.
        """
        return list(self._tools.values())

    async def execute(self, name: str, params: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given parameters.

        Args:
            name: The tool name.
            params: Parameters to pass to the tool.

        Returns:
            ToolResult from the tool execution.
        """
        tool = self.get(name)
        if not tool:
            return ToolResult.from_error(f"Tool '{name}' not found")

        # Validate parameters
        is_valid, error = tool.validate_params(params)
        if not is_valid:
            return ToolResult.from_error(error or "Invalid parameters")

        try:
            logger.info("Executing tool: %s", name)
            result = await tool.execute(**params)
        except Exception as e:  # noqa: BLE001
            logger.exception("Tool %s failed", name)
            return ToolResult.from_error(str(e))
        else:
            logger.info("Tool %s completed with status: %s", name, result.status)
            return result

    def to_anthropic_tools(self, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Convert tools to Anthropic's format.

        Args:
            tool_names: Optional list of tool names to include.
                       If None, includes all tools.

        Returns:
            List of tool definitions in Anthropic's format.
        """
        tools = self._filter_tools(tool_names)
        return [tool.to_anthropic_format() for tool in tools]

    def to_openai_tools(self, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Convert tools to OpenAI's function calling format.

        Args:
            tool_names: Optional list of tool names to include.
                       If None, includes all tools.

        Returns:
            List of tool definitions in OpenAI's format.
        """
        tools = self._filter_tools(tool_names)
        return [tool.to_openai_format() for tool in tools]

    def to_gemini_tools(self, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Convert tools to Google Gemini's format.

        Args:
            tool_names: Optional list of tool names to include.
                       If None, includes all tools.

        Returns:
            List of tool definitions in Gemini's format.
        """
        tools = self._filter_tools(tool_names)
        return [tool.to_gemini_format() for tool in tools]

    def _filter_tools(self, tool_names: list[str] | None) -> list[BaseTool]:
        """Filter tools by names.

        Args:
            tool_names: List of tool names or None for all.

        Returns:
            Filtered list of tools.
        """
        if tool_names is None:
            return self.get_all()
        return [self._tools[name] for name in tool_names if name in self._tools]


# Global tool registry instance
tool_registry = ToolRegistry()
