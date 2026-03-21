"""Built-in tools for the agent system."""

import contextlib
import datetime
import logging
from typing import Any

from agents.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class DateTimeTool(BaseTool):
    """Tool for getting current date and time information."""

    name = "get_datetime"
    description = "Get the current date and time, optionally in a specific timezone."
    parameters = [
        ToolParameter(
            name="timezone",
            type="string",
            description="Timezone name (e.g., 'UTC', 'America/New_York'). Defaults to UTC.",
            required=False,
            default="UTC",
        ),
        ToolParameter(
            name="output_format",
            type="string",
            description="Output format: 'iso' for ISO 8601, 'human' for human-readable.",
            required=False,
            default="iso",
            enum=["iso", "human"],
        ),
    ]

    async def execute(self, timezone: str = "UTC", output_format: str = "iso") -> ToolResult:
        """Get current datetime."""
        try:
            import zoneinfo  # noqa: PLC0415

            tz = zoneinfo.ZoneInfo(timezone)
        except (KeyError, zoneinfo.ZoneInfoNotFoundError):
            tz = datetime.UTC

        now = datetime.datetime.now(tz)

        if output_format == "human":
            output = now.strftime("%A, %B %d, %Y at %I:%M %p %Z")
        else:
            output = now.isoformat()

        return ToolResult.success(
            output=output,
            data={
                "timestamp": now.timestamp(),
                "timezone": str(tz),
                "year": now.year,
                "month": now.month,
                "day": now.day,
                "hour": now.hour,
                "minute": now.minute,
            },
        )


class CalculatorTool(BaseTool):
    """Tool for performing basic mathematical calculations."""

    name = "calculator"
    description = "Perform a mathematical calculation. Supports basic arithmetic operations."
    parameters = [
        ToolParameter(
            name="expression",
            type="string",
            description="Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5', '100 / 4').",
            required=True,
        ),
    ]

    # Allowed operations for safety
    ALLOWED_CHARS = set("0123456789+-*/.() ")

    async def execute(self, expression: str) -> ToolResult:
        """Evaluate a mathematical expression."""
        # Security: Only allow safe characters
        if not all(c in self.ALLOWED_CHARS for c in expression):
            return ToolResult.from_error("Expression contains invalid characters")

        try:
            # Use eval with restricted builtins for safety
            result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
            return ToolResult.success(
                output=str(result),
                data={"expression": expression, "result": result},
            )
        except (SyntaxError, NameError, TypeError, ZeroDivisionError) as e:
            return ToolResult.from_error(f"Calculation error: {e}")


class WebSearchTool(BaseTool):
    """Tool for searching the web (placeholder implementation)."""

    name = "web_search"
    description = "Search the web for information on a given query."
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="The search query.",
            required=True,
        ),
        ToolParameter(
            name="num_results",
            type="number",
            description="Number of results to return (1-10).",
            required=False,
            default=5,
        ),
    ]

    async def execute(self, query: str, num_results: int = 5) -> ToolResult:
        """Search the web (placeholder)."""
        # TODO: Implement actual web search using a search API
        return ToolResult.success(
            output=f"Web search for '{query}' is not yet implemented.",
            data={
                "query": query,
                "num_results": num_results,
                "results": [],
                "note": "This is a placeholder. Configure a search API to enable.",
            },
        )


class BrowserTool(BaseTool):
    """Tool for fetching web page content."""

    name = "browser_fetch"
    description = "Fetch the content of a web page by URL."
    parameters = [
        ToolParameter(
            name="url",
            type="string",
            description="The URL of the web page to fetch.",
            required=True,
        ),
    ]

    async def execute(self, url: str) -> ToolResult:
        """Fetch content from a URL."""
        import urllib.request  # noqa: PLC0415

        try:
            with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
                content = response.read().decode("utf-8", errors="replace")
            return ToolResult.success(
                output=content[:4000],
                data={"url": url, "length": len(content)},
            )
        except Exception as exc:
            return ToolResult.from_error(f"Failed to fetch URL: {exc}")


class MemoryStoreTool(BaseTool):
    """Tool for storing information in the conversation memory."""

    name = "memory_store"
    description = "Store a piece of information in memory for later retrieval."
    parameters = [
        ToolParameter(
            name="key",
            type="string",
            description="A unique key to identify this information.",
            required=True,
        ),
        ToolParameter(
            name="value",
            type="string",
            description="The information to store.",
            required=True,
        ),
    ]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with a memory store."""
        super().__init__(config)
        self._memory: dict[str, str] = {}

    async def execute(self, key: str, value: str) -> ToolResult:
        """Store a value in memory."""
        self._memory[key] = value
        return ToolResult.success(
            output=f"Stored '{key}' in memory.",
            data={"key": key, "stored": True},
        )


class MemoryRetrieveTool(BaseTool):
    """Tool for retrieving information from conversation memory."""

    name = "memory_retrieve"
    description = "Retrieve a previously stored piece of information from memory."
    parameters = [
        ToolParameter(
            name="key",
            type="string",
            description="The key of the information to retrieve.",
            required=True,
        ),
    ]

    def __init__(self, config: dict[str, Any] | None = None, memory: dict[str, str] | None = None) -> None:
        """Initialize with a shared memory store."""
        super().__init__(config)
        self._memory = memory or {}

    async def execute(self, key: str) -> ToolResult:
        """Retrieve a value from memory."""
        if key not in self._memory:
            return ToolResult.from_error(f"Key '{key}' not found in memory.")

        value = self._memory[key]
        return ToolResult.success(
            output=value,
            data={"key": key, "found": True},
        )


class MemorySearchTool(BaseTool):
    """Tool for searching conversation memory using vector + text hybrid search."""

    name = "memory_search"
    description = "Search through past conversation history and summaries to find relevant information."
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="The search query to find relevant memories.",
            required=True,
        ),
        ToolParameter(
            name="search_type",
            type="string",
            description="Type of search: 'hybrid' (combines vector and text), 'vector' (semantic), 'text' (keyword).",
            required=False,
            default="hybrid",
            enum=["hybrid", "vector", "text"],
        ),
        ToolParameter(
            name="max_results",
            type="number",
            description="Maximum number of results to return (1-10).",
            required=False,
            default=5,
        ),
    ]

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        agent_id: int | None = None,
        session_id: int | None = None,
    ) -> None:
        """Initialize with optional agent/session context.

        Args:
            config: Tool configuration.
            agent_id: Optional agent ID to limit search scope.
            session_id: Optional session ID to limit search scope.
        """
        super().__init__(config)
        self.agent_id = agent_id
        self.session_id = session_id

    async def execute(
        self,
        query: str,
        search_type: str = "hybrid",
        max_results: int = 5,
    ) -> ToolResult:
        """Search conversation memory."""
        from memory.models import Session  # noqa: PLC0415
        from memory.search import MemorySearchConfig, MemorySearchService  # noqa: PLC0415

        # Create search service with custom max_results
        config = MemorySearchConfig(max_results=min(max(1, max_results), 10))
        service = MemorySearchService(config)

        # Get session if we have a session_id
        session = None
        if self.session_id:
            with contextlib.suppress(Session.DoesNotExist):
                session = Session.objects.get(id=self.session_id)

        # Perform search
        results = service.search(
            query=query,
            session=session,
            agent_id=self.agent_id,
            search_type=search_type,
        )

        if not results:
            return ToolResult.success(
                output="No relevant memories found.",
                data={"query": query, "results": [], "count": 0},
            )

        # Format results
        formatted = []
        for result in results:
            formatted.append(
                {
                    "content": result.content,
                    "score": round(result.score, 3),
                    "source": result.source,
                    "metadata": result.metadata,
                }
            )

        output_lines = [f"Found {len(results)} relevant memories:"]
        for i, r in enumerate(formatted, 1):
            output_lines.append(f"\n{i}. [{r['source']}] (score: {r['score']})")
            output_lines.append(f"   {r['content'][:200]}...")

        return ToolResult.success(
            output="\n".join(output_lines),
            data={"query": query, "results": formatted, "count": len(results)},
        )


def register_builtin_tools(
    registry: Any,
    agent_id: int | None = None,
    session_id: int | None = None,
) -> None:
    """Register all built-in tools with the given registry.

    Args:
        registry: The ToolRegistry instance.
        agent_id: Optional agent ID for memory search context.
        session_id: Optional session ID for memory search context.
    """
    tools = [
        DateTimeTool(),
        CalculatorTool(),
        WebSearchTool(),
        BrowserTool(),
        MemorySearchTool(agent_id=agent_id, session_id=session_id),
    ]

    for tool in tools:
        try:
            registry.register(tool)
        except ValueError:
            logger.debug("Tool %s already registered", tool.name)
