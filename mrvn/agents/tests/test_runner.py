"""Tests for AgentRunner context bundle injection and MemorySearchTool bundle search."""

import datetime
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from agents.context import CustomerContextBundle, MemoryEntry
from agents.llm import LLMMessage
from agents.runner import AgentRunner
from agents.tools.base import ToolStatus
from agents.tools.builtin import MemorySearchTool


def make_bundle(**kwargs) -> CustomerContextBundle:
    defaults = {
        "customer_id": "acme",
        "claude_md": "# Project Instructions\nBe concise and helpful.",
        "sops": {"onboarding.md": "# Onboarding\nFollow the standard process."},
        "project_goals": "# Goals\nBuild reliable software.",
        "memory_index": "# Memory Index\n- 2024-01-15.md",
        "daily_memories": [],
    }
    defaults.update(kwargs)
    return CustomerContextBundle(**defaults)


def make_memory_entry(content: str, date: datetime.date | None = None) -> MemoryEntry:
    date = date or datetime.date(2024, 1, 15)
    return MemoryEntry(
        date=date,
        filename=f"{date}.md",
        content=content,
        is_weekly_summary=False,
    )


class AgentRunnerContextBundleTests(IsolatedAsyncioTestCase):
    """Tests for AgentRunner with CustomerContextBundle injection."""

    def test_init_accepts_context_bundle(self) -> None:
        bundle = make_bundle()
        runner = AgentRunner(context_bundle=bundle, register_builtins=False)
        self.assertIsNotNone(runner._context_prefix)

    def test_context_prefix_includes_claude_md(self) -> None:
        bundle = make_bundle(claude_md="# My Instructions\nDo this.", sops={}, project_goals="")
        runner = AgentRunner(context_bundle=bundle, register_builtins=False)
        self.assertIn("# My Instructions", runner._context_prefix)

    def test_context_prefix_includes_project_goals(self) -> None:
        bundle = make_bundle(claude_md="", sops={}, project_goals="# Goals\nShip fast.")
        runner = AgentRunner(context_bundle=bundle, register_builtins=False)
        self.assertIn("# Goals", runner._context_prefix)

    def test_context_prefix_includes_sops(self) -> None:
        bundle = make_bundle(claude_md="", sops={"process.md": "# Process\nStep 1."}, project_goals="")
        runner = AgentRunner(context_bundle=bundle, register_builtins=False)
        self.assertIn("process.md", runner._context_prefix)
        self.assertIn("# Process", runner._context_prefix)

    def test_no_bundle_leaves_empty_prefix(self) -> None:
        runner = AgentRunner(register_builtins=False)
        self.assertEqual(runner._context_prefix, "")

    async def test_run_prepends_bundle_context_to_system_prompt(self) -> None:
        bundle = make_bundle(claude_md="BUNDLE_CONTENT", sops={}, project_goals="")
        runner = AgentRunner(context_bundle=bundle, register_builtins=False)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.has_tool_calls = False
        mock_response.content = "Agent reply"
        mock_client.generate = AsyncMock(return_value=mock_response)
        runner._client = mock_client

        await runner.run([LLMMessage.user("Hello")], enable_tools=False)

        mock_client.generate.assert_called_once()
        call_kwargs = mock_client.generate.call_args.kwargs
        self.assertIn("BUNDLE_CONTENT", call_kwargs["system_prompt"])

    async def test_run_prepends_bundle_before_agent_system_prompt(self) -> None:
        bundle = make_bundle(claude_md="BUNDLE", sops={}, project_goals="")
        mock_agent = MagicMock()
        mock_agent.system_prompt = "AGENT_PROMPT"
        mock_agent.rate_limit_enabled = False
        mock_agent.provider = "openai"
        mock_agent.temperature = 0.7
        mock_agent.max_tokens = 4096
        mock_agent.id = 1

        runner = AgentRunner(mock_agent, context_bundle=bundle, register_builtins=False)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.has_tool_calls = False
        mock_response.content = "reply"
        mock_client.generate = AsyncMock(return_value=mock_response)
        runner._client = mock_client

        await runner.run([LLMMessage.user("Hi")], enable_tools=False)

        call_kwargs = mock_client.generate.call_args.kwargs
        prompt = call_kwargs["system_prompt"]
        self.assertIn("BUNDLE", prompt)
        self.assertIn("AGENT_PROMPT", prompt)
        self.assertLess(prompt.index("BUNDLE"), prompt.index("AGENT_PROMPT"))


class MemorySearchToolBundleTests(IsolatedAsyncioTestCase):
    """Tests for MemorySearchTool bundle memory search."""

    def _make_memories(self) -> list[MemoryEntry]:
        return [
            make_memory_entry("Met with Alice to discuss project roadmap and Q1 planning."),
            make_memory_entry("Resolved bug in the authentication module.", datetime.date(2024, 1, 16)),
            MemoryEntry(
                date=datetime.date(2024, 1, 20),
                filename="2024-01-20_WEEKLY_SUMMARY.md",
                content="Weekly summary: completed setup and first deployment.",
                is_weekly_summary=True,
            ),
        ]

    async def test_finds_matching_memory(self) -> None:
        tool = MemorySearchTool(bundle_memories=self._make_memories())
        result = await tool.execute(query="roadmap")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.data["count"], 1)
        self.assertIn("roadmap", result.data["results"][0]["content"].lower())

    async def test_no_match_returns_empty(self) -> None:
        tool = MemorySearchTool(bundle_memories=self._make_memories())
        result = await tool.execute(query="xyzzy_no_match_here")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.data["count"], 0)

    async def test_result_source_prefixed_with_bundle(self) -> None:
        tool = MemorySearchTool(bundle_memories=self._make_memories())
        result = await tool.execute(query="authentication")
        self.assertTrue(result.data["results"][0]["source"].startswith("bundle:"))

    async def test_respects_max_results(self) -> None:
        memories = [make_memory_entry(f"Note about topic number {i}.") for i in range(8)]
        tool = MemorySearchTool(bundle_memories=memories)
        result = await tool.execute(query="topic", max_results=3)
        self.assertLessEqual(result.data["count"], 3)

    async def test_bundle_bypasses_db(self) -> None:
        """Bundle search must not import or call DB-backed MemorySearchService."""
        tool = MemorySearchTool(bundle_memories=self._make_memories())
        with (
            patch(
                "agents.tools.builtin.MemorySearchTool._search_bundle_memories",
                wraps=tool._search_bundle_memories,
            ) as mock_bundle_search,
            patch.dict("sys.modules", {"memory.search": None}),
        ):
            result = await tool.execute(query="roadmap")
        mock_bundle_search.assert_called_once()
        self.assertEqual(result.status, ToolStatus.SUCCESS)

    async def test_no_bundle_memories_falls_through_to_db_path(self) -> None:
        """Without bundle memories, execute() should attempt DB search."""
        tool = MemorySearchTool()
        with (
            patch("memory.models.Session"),
            patch("memory.search.MemorySearchConfig"),
            patch("memory.search.MemorySearchService") as mock_service_cls,
        ):
            mock_service = MagicMock()
            mock_service.search.return_value = []
            mock_service_cls.return_value = mock_service
            result = await tool.execute(query="test")
        mock_service_cls.assert_called_once()
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.data["count"], 0)
