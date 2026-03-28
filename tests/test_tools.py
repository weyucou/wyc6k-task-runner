"""Tests for marvin tool validation and execution."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marvin.tools.builtin import (
    BrowserTool,
    CalculatorTool,
    DateTimeTool,
    MemorySearchTool,
    WebSearchTool,
)
from marvin.tools.coding import GitHubIssueTool, GitHubPRTool
from marvin.tools.registry import ToolRegistry

# ---- CalculatorTool ----


class TestCalculatorToolValidation:
    def setup_method(self) -> None:
        self.tool = CalculatorTool()

    def test_missing_required_parameter(self) -> None:
        is_valid, error = self.tool.validate_params({})
        assert not is_valid
        assert "expression" in error

    def test_wrong_type_number_for_string(self) -> None:
        is_valid, error = self.tool.validate_params({"expression": 123})
        assert not is_valid
        assert "string" in error.lower()

    def test_valid_string_parameter(self) -> None:
        is_valid, error = self.tool.validate_params({"expression": "1 + 1"})
        assert is_valid
        assert error is None

    def test_extra_parameters_accepted(self) -> None:
        is_valid, error = self.tool.validate_params({"expression": "1 + 1", "unknown_param": "value"})
        assert is_valid
        assert error is None

    def test_execute_valid_expression(self) -> None:
        result = asyncio.run(self.tool.execute(expression="2 + 3"))
        assert result.output == "5"

    def test_execute_invalid_expression(self) -> None:
        result = asyncio.run(self.tool.execute(expression="1 / 0"))
        assert result.error is not None

    def test_execute_disallowed_chars(self) -> None:
        result = asyncio.run(self.tool.execute(expression="__import__('os')"))
        assert result.error is not None


# ---- DateTimeTool ----


class TestDateTimeToolValidation:
    def setup_method(self) -> None:
        self.tool = DateTimeTool()

    def test_no_parameters_valid(self) -> None:
        is_valid, error = self.tool.validate_params({})
        assert is_valid
        assert error is None

    def test_valid_timezone_parameter(self) -> None:
        is_valid, error = self.tool.validate_params({"timezone": "America/New_York"})
        assert is_valid
        assert error is None

    def test_valid_output_format_iso(self) -> None:
        is_valid, error = self.tool.validate_params({"output_format": "iso"})
        assert is_valid

    def test_valid_output_format_human(self) -> None:
        is_valid, error = self.tool.validate_params({"output_format": "human"})
        assert is_valid

    def test_invalid_output_format_enum(self) -> None:
        is_valid, error = self.tool.validate_params({"output_format": "invalid"})
        assert not is_valid
        assert "must be one of" in error.lower()

    def test_wrong_type_for_timezone(self) -> None:
        is_valid, error = self.tool.validate_params({"timezone": 123})
        assert not is_valid
        assert "string" in error.lower()


# ---- WebSearchTool ----


class TestWebSearchToolValidation:
    def setup_method(self) -> None:
        self.tool = WebSearchTool()

    def test_missing_required_query(self) -> None:
        is_valid, error = self.tool.validate_params({})
        assert not is_valid
        assert "query" in error

    def test_valid_query_only(self) -> None:
        is_valid, error = self.tool.validate_params({"query": "test search"})
        assert is_valid
        assert error is None

    def test_wrong_type_for_num_results(self) -> None:
        is_valid, error = self.tool.validate_params({"query": "test", "num_results": "five"})
        assert not is_valid
        assert "number" in error.lower()


# ---- BrowserTool ----


class TestBrowserToolValidation:
    def setup_method(self) -> None:
        self.tool = BrowserTool()

    def test_missing_required_url(self) -> None:
        is_valid, error = self.tool.validate_params({})
        assert not is_valid
        assert "url" in error

    def test_valid_url_parameter(self) -> None:
        is_valid, error = self.tool.validate_params({"url": "http://example.com"})
        assert is_valid
        assert error is None


# ---- MemorySearchTool (stateless) ----


class TestMemorySearchToolStateless:
    def test_returns_empty_without_session(self) -> None:
        tool = MemorySearchTool()
        result = asyncio.run(tool.execute(query="test"))
        assert result.output == "Memory search not available in stateless mode."
        assert result.data["count"] == 0

    def test_returns_empty_with_session_id(self) -> None:
        tool = MemorySearchTool(session_id="some-session")
        result = asyncio.run(tool.execute(query="test"))
        assert result.output == "Memory search not available in stateless mode."
        assert result.data["results"] == []


# ---- ToolSchemaGeneration ----


class TestToolSchemaGeneration:
    def test_calculator_schema_has_required_fields(self) -> None:
        tool = CalculatorTool()
        schema = tool.get_schema()
        assert schema["type"] == "object"
        assert "expression" in schema["properties"]
        assert "expression" in schema["required"]

    def test_datetime_schema_has_enum(self) -> None:
        tool = DateTimeTool()
        schema = tool.get_schema()
        assert "output_format" in schema["properties"]
        assert "enum" in schema["properties"]["output_format"]
        assert schema["properties"]["output_format"]["enum"] == ["iso", "human"]

    def test_datetime_schema_no_required_fields(self) -> None:
        tool = DateTimeTool()
        schema = tool.get_schema()
        assert len(schema.get("required", [])) == 0

    def test_web_search_schema_has_default(self) -> None:
        tool = WebSearchTool()
        schema = tool.get_schema()
        assert "num_results" in schema["properties"]
        assert schema["properties"]["num_results"]["default"] == 5


# ---- ToolRegistry ----


class TestToolRegistry:
    def test_register_and_list(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        assert "calculator" in registry.list_tools()

    def test_duplicate_registration_raises(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(CalculatorTool())

    def test_unregister(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.unregister("calculator")
        assert "calculator" not in registry.list_tools()

    def test_execute_unknown_tool_returns_error(self) -> None:
        registry = ToolRegistry()
        result = asyncio.run(registry.execute("no_such_tool", {}))
        assert result.error is not None
        assert "not found" in result.error

    def test_to_anthropic_tools_format(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        tools = registry.to_anthropic_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "calculator"
        assert "input_schema" in tools[0]

    def test_to_openai_tools_format(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        tools = registry.to_openai_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "calculator"


# ---- GitHubIssueTool ----


class TestGitHubIssueTool:
    def setup_method(self) -> None:
        self.tool = GitHubIssueTool()

    def test_require_approval_is_true(self) -> None:
        assert self.tool.require_approval is True

    def test_missing_required_action(self) -> None:
        is_valid, error = self.tool.validate_params({"issue_url": "https://github.com/o/r/issues/1"})
        assert not is_valid
        assert "action" in error

    def test_missing_required_issue_url(self) -> None:
        is_valid, error = self.tool.validate_params({"action": "view"})
        assert not is_valid
        assert "issue_url" in error

    def test_invalid_action_enum(self) -> None:
        is_valid, error = self.tool.validate_params(
            {"action": "delete", "issue_url": "https://github.com/o/r/issues/1"}
        )
        assert not is_valid
        assert "must be one of" in error.lower()

    def test_valid_view_params(self) -> None:
        is_valid, error = self.tool.validate_params({"action": "view", "issue_url": "https://github.com/o/r/issues/1"})
        assert is_valid
        assert error is None

    def test_missing_token_returns_error(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = asyncio.run(self.tool.execute(action="view", issue_url="https://github.com/o/r/issues/1"))
        assert result.error == "GITHUB_TOKEN environment variable is not set"

    def test_missing_gh_cli_returns_error(self) -> None:
        with patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}):
            with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
                result = asyncio.run(self.tool.execute(action="view", issue_url="https://github.com/o/r/issues/1"))
        assert result.error is not None
        assert "'gh' CLI not found" in result.error

    def test_comment_missing_body_returns_error(self) -> None:
        with patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}):
            result = asyncio.run(
                self.tool.execute(action="comment", issue_url="https://github.com/o/r/issues/1", body="")
            )
        assert result.error is not None
        assert "body is required" in result.error

    def test_add_label_missing_label_returns_error(self) -> None:
        with patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}):
            result = asyncio.run(
                self.tool.execute(action="add_label", issue_url="https://github.com/o/r/issues/1", label="")
            )
        assert result.error is not None
        assert "label is required" in result.error

    def test_success_view(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"issue output", b""))
        with patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}):
            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = asyncio.run(self.tool.execute(action="view", issue_url="https://github.com/o/r/issues/1"))
        assert result.error is None
        assert "issue output" in result.output

    def test_gh_nonzero_exit_returns_error(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"not found"))
        with patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}):
            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = asyncio.run(self.tool.execute(action="view", issue_url="https://github.com/o/r/issues/1"))
        assert result.error is not None
        assert "gh exited with code 1" in result.error


# ---- GitHubPRTool ----


class TestGitHubPRTool:
    def setup_method(self) -> None:
        self.tool = GitHubPRTool()

    def test_require_approval_is_true(self) -> None:
        assert self.tool.require_approval is True

    def test_missing_required_action(self) -> None:
        is_valid, error = self.tool.validate_params({"pr_url": "https://github.com/o/r/pull/1"})
        assert not is_valid
        assert "action" in error

    def test_invalid_action_enum(self) -> None:
        is_valid, error = self.tool.validate_params({"action": "merge", "pr_url": "https://github.com/o/r/pull/1"})
        assert not is_valid
        assert "must be one of" in error.lower()

    def test_valid_view_params(self) -> None:
        is_valid, error = self.tool.validate_params({"action": "view", "pr_url": "https://github.com/o/r/pull/1"})
        assert is_valid
        assert error is None

    def test_missing_token_returns_error(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            result = asyncio.run(self.tool.execute(action="view", pr_url="https://github.com/o/r/pull/1"))
        assert result.error == "GITHUB_TOKEN environment variable is not set"

    def test_missing_gh_cli_returns_error(self) -> None:
        with patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}):
            with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
                result = asyncio.run(self.tool.execute(action="view", pr_url="https://github.com/o/r/pull/1"))
        assert result.error is not None
        assert "'gh' CLI not found" in result.error

    def test_create_missing_title_returns_error(self) -> None:
        with patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}):
            result = asyncio.run(self.tool.execute(action="create", title=""))
        assert result.error is not None
        assert "title is required" in result.error

    def test_view_missing_pr_url_returns_error(self) -> None:
        with patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}):
            result = asyncio.run(self.tool.execute(action="view", pr_url=""))
        assert result.error is not None
        assert "pr_url is required" in result.error

    def test_update_body_missing_pr_url_returns_error(self) -> None:
        with patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}):
            result = asyncio.run(self.tool.execute(action="update_body", pr_url=""))
        assert result.error is not None
        assert "pr_url is required" in result.error

    def test_success_view(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"pr output", b""))
        with patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}):
            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = asyncio.run(self.tool.execute(action="view", pr_url="https://github.com/o/r/pull/1"))
        assert result.error is None
        assert "pr output" in result.output

    def test_gh_nonzero_exit_returns_error(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error msg"))
        with patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}):
            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = asyncio.run(self.tool.execute(action="view", pr_url="https://github.com/o/r/pull/1"))
        assert result.error is not None
        assert "gh exited with code 1" in result.error
