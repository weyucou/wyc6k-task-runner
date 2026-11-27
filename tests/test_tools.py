# Copyright (c) 2026 MNKT
"""Tests for marvin tool validation and execution."""

import asyncio
import base64
import http.client
import inspect
import io
import os
import tempfile
import threading
import time
import tracemalloc
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import pytest

from marvin.settings import TOOL_OUTPUT_MAX_BYTES, TOOL_OUTPUT_MAX_LINES
from marvin.tools import coding, truncate
from marvin.tools.base import ToolResult, ToolStatus
from marvin.tools.builtin import (
    BrowserTool,
    CalculatorTool,
    DateTimeTool,
    MemorySearchTool,
    WebSearchTool,
)
from marvin.tools.coding import (
    _BACKGROUND_PROCESSES,
    AskccRunTool,
    BrowserAutomationTool,
    EditTool,
    ExecTool,
    GitHubIssueTool,
    GitHubPRTool,
    ImageTool,
    ProcessTool,
    ReadTool,
    WebFetchTool,
    WriteTool,
)
from marvin.tools.registry import ToolRegistry
from marvin.tools.truncate import FETCH_MAX_BYTES, measure
from tests.notices import HEAD_NOTICE, TAIL_NOTICE

if TYPE_CHECKING:
    from collections.abc import Iterator

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


# ---- WebSearchTool execution ----


class TestWebSearchTool:
    def setup_method(self) -> None:
        self.tool = WebSearchTool()

    def test_unconfigured_returns_error(self) -> None:
        with patch("marvin.tools.builtin.ZAATAR_SEARCH_API_URL", ""):
            result = asyncio.run(self.tool.execute(query="python"))
        assert result.error is not None
        assert "not configured" in result.error

    def test_successful_search_returns_results(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": "Python.org", "url": "https://www.python.org/", "description": "Official Python site"},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("marvin.tools.builtin.ZAATAR_SEARCH_API_URL", "http://localhost:5000"),
            patch("httpx.AsyncClient", return_value=mock_cm),
        ):
            result = asyncio.run(self.tool.execute(query="python"))

        assert result.error is None
        assert result.data["count"] == 1
        assert result.data["results"][0]["title"] == "Python.org"
        assert result.data["results"][0]["snippet"] == "Official Python site"

    def test_http_error_returns_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPError("Connection refused"))

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("marvin.tools.builtin.ZAATAR_SEARCH_API_URL", "http://localhost:5000"),
            patch("httpx.AsyncClient", return_value=mock_cm),
        ):
            result = asyncio.run(self.tool.execute(query="python"))

        assert result.error is not None
        assert "Search API request failed" in result.error

    def test_num_results_forwarded_as_count(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("marvin.tools.builtin.ZAATAR_SEARCH_API_URL", "http://localhost:5000"),
            patch("httpx.AsyncClient", return_value=mock_cm),
        ):
            asyncio.run(self.tool.execute(query="python", num_results=3))

        call_kwargs = mock_client.get.call_args
        assert call_kwargs.kwargs["params"]["count"] == 3


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


class TestBrowserTool:
    def test_fetches_off_the_event_loop_thread(self) -> None:
        calls: list[tuple[str, int, threading.Thread]] = []

        def fake_safe_urlopen(url: str, timeout: int) -> io.BytesIO:
            calls.append((url, timeout, threading.current_thread()))
            return io.BytesIO(b"<p>hello</p>")

        with patch("marvin.tools.builtin.safe_urlopen", side_effect=fake_safe_urlopen):
            result = asyncio.run(BrowserTool().execute(url="http://example.com"))
        assert result.error is None
        assert result.output == "<p>hello</p>"
        assert result.data == {"url": "http://example.com", "length": 12}
        [(url, timeout, fetch_thread)] = calls
        assert (url, timeout) == ("http://example.com", 10)
        assert fetch_thread is not threading.main_thread()

    @pytest.mark.parametrize(
        "exc",
        [
            urllib.error.URLError("connection refused"),
            TimeoutError("timed out"),
            ValueError("unknown url type: 'example'"),
            http.client.IncompleteRead(b""),
        ],
        ids=["url-error", "timeout", "bad-url", "http-protocol"],
    )
    def test_fetch_errors_return_an_error_result(self, exc) -> None:
        with patch("marvin.tools.builtin.safe_urlopen", side_effect=exc):
            result = asyncio.run(BrowserTool().execute(url="http://example.com"))
        assert result.error is not None
        assert result.error.startswith("Failed to fetch URL:")


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


# ---- ToolResult ----


def _within_the_cap(content: str) -> bool:
    lines, size = measure(content)
    return lines <= TOOL_OUTPUT_MAX_LINES and size <= TOOL_OUTPUT_MAX_BYTES


class TestToolResult:
    @pytest.mark.parametrize(
        ("result", "content"),
        [
            (ToolResult.success("4"), "4"),
            (ToolResult.from_error("boom"), "Error: boom"),
            (ToolResult.from_error("boom", output="partial"), "partial\n\nError: boom"),
            (ToolResult(status=ToolStatus.ERROR, output=""), "Error: Tool failed"),
        ],
        ids=["success", "error", "error-after-partial-output", "error-without-message"],
    )
    def test_to_model_content(self, result: ToolResult, content: str) -> None:
        assert result.to_model_content() == content

    def test_is_error(self) -> None:
        assert ToolResult.from_error("boom").is_error
        assert not ToolResult.success("4").is_error

    def test_an_oversized_result_is_capped(self) -> None:
        content = ToolResult.success("x\n" * 200_000).to_model_content()
        assert _within_the_cap(content)
        assert content.startswith("x\nx\n")
        assert HEAD_NOTICE.fullmatch(content.rsplit("\n", 1)[1])

    def test_an_oversized_error_result_keeps_its_error_after_the_notice(self) -> None:
        result = ToolResult.from_error("Command exited with code 1.", output="x\n" * 200_000)
        content = result.to_model_content()
        assert _within_the_cap(content)
        *_, notice, blank, error = content.split("\n")
        assert HEAD_NOTICE.fullmatch(notice)
        assert (blank, error) == ("", "Error: Command exited with code 1.")

    def test_an_oversized_error_message_is_capped_too(self) -> None:
        content = ToolResult.from_error("y\n" * 200_000, output="x\n" * 200_000).to_model_content()
        assert _within_the_cap(content)
        output, error = content.split("\n\nError: ")
        assert HEAD_NOTICE.fullmatch(output.rsplit("\n", 1)[1])
        assert error.startswith("y\ny\n")
        assert HEAD_NOTICE.fullmatch(error.rsplit("\n", 1)[1])

    def test_an_oversized_error_without_output_is_capped(self) -> None:
        content = ToolResult.from_error("y\n" * 200_000).to_model_content()
        assert _within_the_cap(content)
        assert content.startswith("Error: y\ny\n")

    def test_a_long_error_is_kept_whole_while_everything_fits(self) -> None:
        error = "e" * (TOOL_OUTPUT_MAX_BYTES * 3 // 4)  # over half the cap
        result = ToolResult.from_error(error, output="partial")
        assert result.to_model_content() == f"partial\n\nError: {error}"


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

    def test_require_approval_is_false(self) -> None:
        assert self.tool.require_approval is False

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
        with (
            patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}),
            patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError),
        ):
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
        with (
            patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = asyncio.run(self.tool.execute(action="view", issue_url="https://github.com/o/r/issues/1"))
        assert result.error is None
        assert "issue output" in result.output

    def test_gh_nonzero_exit_returns_error(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"not found"))
        with (
            patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = asyncio.run(self.tool.execute(action="view", issue_url="https://github.com/o/r/issues/1"))
        assert result.error is not None
        assert "gh exited with code 1" in result.error


# ---- GitHubPRTool ----


class TestGitHubPRTool:
    def setup_method(self) -> None:
        self.tool = GitHubPRTool()

    def test_require_approval_is_false(self) -> None:
        assert self.tool.require_approval is False

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
        with (
            patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}),
            patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError),
        ):
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
        with (
            patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = asyncio.run(self.tool.execute(action="view", pr_url="https://github.com/o/r/pull/1"))
        assert result.error is None
        assert "pr output" in result.output

    def test_gh_nonzero_exit_returns_error(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error msg"))
        with (
            patch.dict("os.environ", {"GITHUB_TOKEN": "tok"}),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            result = asyncio.run(self.tool.execute(action="view", pr_url="https://github.com/o/r/pull/1"))
        assert result.error is not None
        assert "gh exited with code 1" in result.error


# ---- AskccRunTool ----


class TestAskccRunTool:
    def setup_method(self) -> None:
        self.tool = AskccRunTool()

    def test_require_approval_is_true(self) -> None:
        assert self.tool.require_approval is True

    def test_missing_required_action(self) -> None:
        is_valid, error = self.tool.validate_params({"issue_url": "https://github.com/o/r/issues/1"})
        assert not is_valid
        assert "action" in error

    def test_missing_required_issue_url(self) -> None:
        is_valid, error = self.tool.validate_params({"action": "prepare"})
        assert not is_valid
        assert "issue_url" in error

    def test_invalid_action_enum(self) -> None:
        is_valid, error = self.tool.validate_params(
            {"action": "deploy", "issue_url": "https://github.com/o/r/issues/1"}
        )
        assert not is_valid
        assert "must be one of" in error.lower()

    def test_valid_params_each_action(self) -> None:
        for action in ("prepare", "develop", "review", "diagnose", "explore", "plan"):
            is_valid, error = self.tool.validate_params(
                {"action": action, "issue_url": "https://github.com/o/r/issues/1"}
            )
            assert is_valid, f"action={action} should be valid, got error: {error}"
            assert error is None

    def test_missing_askcc_cli_returns_error(self) -> None:
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = asyncio.run(self.tool.execute(action="prepare", issue_url="https://github.com/o/r/issues/1"))
        assert result.error is not None
        assert "'askcc' CLI not found" in result.error

    def test_nonzero_exit_returns_error(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"something went wrong"))
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = asyncio.run(self.tool.execute(action="develop", issue_url="https://github.com/o/r/issues/1"))
        assert result.error is not None
        assert "askcc exited with code 1" in result.error

    def test_success(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"done", b""))
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = asyncio.run(self.tool.execute(action="plan", issue_url="https://github.com/o/r/issues/1"))
        assert result.error is None
        assert "done" in result.output

    def test_timeout_terminates_process_and_returns_error(self) -> None:
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.communicate = AsyncMock(side_effect=TimeoutError)
        mock_proc.wait = AsyncMock(return_value=None)
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = asyncio.run(self.tool.execute(action="develop", issue_url="https://github.com/o/r/issues/1"))
        mock_proc.terminate.assert_called_once()
        assert result.error is not None
        assert "timed out" in result.error

    def test_config_timeout_override(self) -> None:
        custom_timeout = 600
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))
        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("asyncio.wait_for", wraps=asyncio.wait_for) as mock_wait,
        ):
            asyncio.run(
                self.tool.execute(
                    action="review",
                    issue_url="https://github.com/o/r/issues/1",
                    config={"timeout": custom_timeout},
                )
            )
        _, kwargs = mock_wait.call_args
        assert kwargs.get("timeout") == custom_timeout


# ---- ImageTool ----


class TestImageToolVision:
    def test_call_vision_returns_text_after_non_text_block(self) -> None:
        reply = SimpleNamespace(content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="A cat.")])
        sdk_create = anthropic.Anthropic(api_key="x").messages.create
        with patch("anthropic.Anthropic") as client_cls:
            create = client_cls.return_value.messages.create
            create.return_value = reply
            result = ImageTool()._call_vision("key", "aGk=", "image/png", "Describe this image.")
        assert result == "A cat."
        inspect.signature(sdk_create).bind(**create.call_args.kwargs)  # request fits the installed SDK

    def test_load_image_reads_a_local_file_under_home(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        raw = b"\x89PNG\r\n\x1a\n" + bytes(8)
        (tmp_path / "pic.png").write_bytes(raw)
        encoded, media_type = asyncio.run(ImageTool()._load_image("~/pic.png"))
        assert base64.b64decode(encoded) == raw
        assert media_type == "image/png"


# ---- File tools: read / write / edit ----


class TestFileTools:
    def test_read_expands_home_and_slices_lines(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
        result = asyncio.run(ReadTool().execute(path="~/notes.txt", offset=2, limit=1))
        # A limit inside the cap gets the offset hint only.
        assert result.output == "two\n\n[Showing lines 2-2 of 3. Use offset=3 to continue.]"
        assert result.data == {
            "path": "~/notes.txt",
            "lines_read": 1,
            "total_lines": 3,
            "next_offset": 3,
            "truncated": False,
        }

    def test_read_a_small_file_whole(self, tmp_path) -> None:
        target = tmp_path / "small.txt"
        target.write_text("one\ntwo", encoding="utf-8")
        result = asyncio.run(ReadTool().execute(path=str(target)))
        assert result.output == "one\ntwo"
        assert (result.data["next_offset"], result.data["truncated"]) == (None, False)

    @pytest.mark.parametrize("line", ["{n}", "{n:0100}"], ids=["over-the-lines", "over-the-bytes"])
    def test_read_a_big_file_shows_the_head_and_the_offset_to_continue(self, tmp_path, line) -> None:
        target = tmp_path / "big.txt"
        target.write_text("".join(line.format(n=n) + "\n" for n in range(1, 100_001)), encoding="utf-8")
        result = asyncio.run(ReadTool().execute(path=str(target)))
        assert _within_the_cap(result.output)
        *lines, blank, notice = result.output.split("\n")
        last = len(lines)
        assert lines == [line.format(n=n) for n in range(1, last + 1)]
        assert blank == ""
        assert notice == f"[Showing lines 1-{last} of 100000. Use offset={last + 1} to continue.]"
        assert (result.data["next_offset"], result.data["truncated"]) == (last + 1, True)

        following = asyncio.run(ReadTool().execute(path=str(target), offset=last + 1))
        assert following.output.startswith(line.format(n=last + 1) + "\n")

    def test_read_cuts_a_long_line(self, tmp_path) -> None:
        target = tmp_path / "long.txt"
        target.write_text("a" * 100_000 + "\nshort\n" + "b" * 2001, encoding="utf-8")
        result = asyncio.run(ReadTool().execute(path=str(target)))
        assert result.output == (
            "a" * 2000 + "… [line truncated, 100000 chars]\nshort\n" + "b" * 2000 + "… [line truncated, 2001 chars]"
        )
        assert result.data["total_lines"] == 3

    def test_read_keeps_memory_bounded(self, tmp_path) -> None:
        target = tmp_path / "one-line.txt"
        target.write_text("a" * (10 * 2**20) + "\nshort\n", encoding="utf-8")
        tracemalloc.start()
        try:
            result = asyncio.run(ReadTool().execute(path=str(target)))
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 4 * 2**20  # the file holds 10 MiB
        assert result.output.endswith(f"… [line truncated, {10 * 2**20} chars]\nshort\n")

    def test_read_missing_file_returns_error(self, tmp_path) -> None:
        missing = tmp_path / "missing.txt"
        result = asyncio.run(ReadTool().execute(path=str(missing)))
        assert result.error == f"File not found: {missing}"

    @pytest.mark.parametrize("path", ["out.txt", "a/b/out.txt"])
    def test_write_creates_missing_parent_directories(self, tmp_path, monkeypatch, path) -> None:
        monkeypatch.chdir(tmp_path)
        result = asyncio.run(WriteTool().execute(path=path, content="héllo"))
        assert result.error is None
        assert result.data == {"path": path, "bytes_written": 6}
        assert (tmp_path / path).read_text(encoding="utf-8") == "héllo"

    def test_edit_replaces_a_unique_string(self, tmp_path) -> None:
        target = tmp_path / "code.py"
        target.write_text("a = 1\nb = 2\n", encoding="utf-8")
        result = asyncio.run(EditTool().execute(path=str(target), old_string="b = 2", new_string="b = 3"))
        assert result.error is None
        assert target.read_text(encoding="utf-8") == "a = 1\nb = 3\n"

    @pytest.mark.parametrize(("old_string", "error"), [("c = 3", "String not found"), (" = ", "appears 2 times")])
    def test_edit_rejects_a_missing_or_ambiguous_string(self, tmp_path, old_string, error) -> None:
        target = tmp_path / "code.py"
        target.write_text("a = 1\nb = 2\n", encoding="utf-8")
        result = asyncio.run(EditTool().execute(path=str(target), old_string=old_string, new_string="x"))
        assert error in result.error
        assert target.read_text(encoding="utf-8") == "a = 1\nb = 2\n"


# ---- ExecTool / ProcessTool ----


def _alive(pid: int) -> bool:
    """True while pid runs; a zombie (exited, not yet reaped by its parent) counts as dead."""
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except ProcessLookupError, FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _dies_within(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while _alive(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def _read_pid(path: Path) -> int:
    """Wait for a background shell's `echo $! > path` to land, then return the pid."""
    deadline = time.monotonic() + 5
    while not (path.exists() and path.read_text(encoding="utf-8").endswith("\n")):
        assert time.monotonic() < deadline, f"{path} was never written"
        time.sleep(0.01)
    return int(path.read_text(encoding="utf-8"))


@pytest.fixture
def blocking_stdin() -> Iterator[None]:
    """Point fd 0 at an empty pipe that stays open, so a child that inherits stdin blocks reading it."""
    read_fd, write_fd = os.pipe()
    saved_stdin = os.dup(0)
    os.dup2(read_fd, 0)
    yield
    os.dup2(saved_stdin, 0)
    for fd in (saved_stdin, read_fd, write_fd):
        os.close(fd)


class TestExecTool:
    def test_runs_in_the_expanded_cwd(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        result = asyncio.run(ExecTool().execute(command="pwd -P", cwd="~"))
        assert result.error is None
        assert result.output.strip() == str(tmp_path.resolve())

    def test_timeout_kills_the_process_group(self, tmp_path) -> None:
        start = time.monotonic()
        result = asyncio.run(ExecTool().execute(command="sleep 30 & echo $! > pid; wait", cwd=str(tmp_path), timeout=1))
        assert time.monotonic() - start < 3
        assert "timed out" in result.error
        assert "larger timeout" in result.error
        assert result.to_model_content().count(result.error) == 1  # the model sees the message once
        assert not _alive(int((tmp_path / "pid").read_text(encoding="utf-8")))

    def test_timeout_escalates_to_sigkill(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(coding, "_KILL_GRACE_SECONDS", 0.2)
        start = time.monotonic()
        result = asyncio.run(
            ExecTool().execute(command="trap '' TERM; sleep 30 & echo $! > pid; wait", cwd=str(tmp_path), timeout=1)
        )
        assert "timed out" in result.error
        assert _dies_within(int((tmp_path / "pid").read_text(encoding="utf-8")), 0.5)
        assert time.monotonic() - start < 3

    def test_cancel_kills_the_process_group(self, tmp_path) -> None:
        command = ExecTool().execute(command="sleep 30 & echo $! > pid; wait", cwd=str(tmp_path), timeout=60)
        with pytest.raises(TimeoutError):  # wait_for cancels the tool call
            asyncio.run(asyncio.wait_for(command, 0.5))
        assert not _alive(_read_pid(tmp_path / "pid"))

    def test_stdin_is_closed(self, blocking_stdin) -> None:
        start = time.monotonic()
        result = asyncio.run(ExecTool().execute(command="cat", timeout=10))
        assert time.monotonic() - start < 2
        assert result.data.get("return_code") == 0

    def test_backgrounded_child_does_not_hang(self, tmp_path) -> None:
        start = time.monotonic()
        result = asyncio.run(
            ExecTool().execute(command="sleep 30 & echo $! > pid; echo done", cwd=str(tmp_path), timeout=10)
        )
        assert time.monotonic() - start < 2
        assert result.data.get("return_code") == 0
        assert result.output.startswith("done\n")
        assert "`process` tool" in result.output
        assert not _alive(int((tmp_path / "pid").read_text(encoding="utf-8")))

    def test_exit_code_preserved_with_background_child(self) -> None:
        start = time.monotonic()
        result = asyncio.run(ExecTool().execute(command="sleep 30 & exit 3", timeout=10))
        assert time.monotonic() - start < 2
        assert result.data.get("return_code") == 3
        assert result.error == "Command exited with code 3."

    def test_default_timeout_is_minutes(self) -> None:
        [param] = [p for p in ExecTool.parameters if p.name == "timeout"]
        assert param.default == 120
        assert "larger" in param.description
        assert inspect.signature(ExecTool.execute).parameters["timeout"].default == 120

    def test_small_output_is_whole_and_creates_no_spill_dir(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        result = asyncio.run(ExecTool().execute(command="echo hi"))
        assert result.output == "hi\n"
        assert result.data == {"return_code": 0, "truncated": False, "spill_path": None}
        assert list(tmp_path.iterdir()) == []

    def test_stderr_is_interleaved_with_stdout(self) -> None:
        result = asyncio.run(ExecTool().execute(command="echo out; echo err >&2; echo more"))
        assert result.output == "out\nerr\nmore\n"

    def test_long_output_keeps_the_tail_and_spills_all_of_it(self) -> None:
        result = asyncio.run(ExecTool().execute(command="seq 1 100000"))
        assert _within_the_cap(result.to_model_content())
        *tail, blank, notice = result.output.split("\n")
        match = TAIL_NOTICE.fullmatch(notice)
        assert match is not None
        assert blank == ""
        assert tail == [str(n) for n in range(100_001 - int(match[1]), 100_001)]
        assert int(match[2]) == 100_000
        assert Path(match[5]).read_text(encoding="utf-8") == "".join(f"{n}\n" for n in range(1, 100_001))
        assert result.data == {"return_code": 0, "truncated": True, "spill_path": match[5]}

    @pytest.mark.parametrize(
        ("command", "timeout", "error"),
        [
            ("seq 1 100000; exit 3", 10, "Command exited with code 3."),
            ("seq 1 100000; exec sleep 30", 1, "Command timed out after 1s"),
        ],
        ids=["non-zero-exit", "timeout"],
    )
    def test_a_failed_command_keeps_the_tail_notice_and_the_error(self, command, timeout, error) -> None:
        content = asyncio.run(ExecTool().execute(command=command, timeout=timeout)).to_model_content()
        assert _within_the_cap(content)
        *_, notice, blank, last = content.split("\n")
        assert TAIL_NOTICE.fullmatch(notice)
        assert blank == ""
        assert last.startswith(f"Error: {error}")

    def test_memory_stays_bounded_for_a_flood_of_output(self) -> None:
        tracemalloc.start()
        try:
            result = asyncio.run(ExecTool().execute(command="yes | head -c 50000000"))
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 8 * 2**20  # the command prints 50 MB
        assert Path(result.data["spill_path"]).stat().st_size == 50_000_000


class TestProcessTool:
    def test_output_is_readable_while_the_process_runs(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        tool = ProcessTool()
        started = asyncio.run(
            tool.execute(action="start", session_id="stream", command="pwd -P; exec sleep 30", cwd="~")
        )
        assert started.error is None
        proc = _BACKGROUND_PROCESSES["stream"]["proc"]
        try:
            output = ""
            deadline = time.monotonic() + 5
            while not output and time.monotonic() < deadline:
                time.sleep(0.05)
                output = asyncio.run(tool.execute(action="output", session_id="stream")).output
            assert output == f"{tmp_path.resolve()}\n"
            assert asyncio.run(tool.execute(action="status", session_id="stream")).data["running"] is True
        finally:
            asyncio.run(tool.execute(action="stop", session_id="stream"))
            proc.wait(timeout=5)

    def test_output_returns_only_new_lines(self) -> None:
        tool = ProcessTool()
        asyncio.run(tool.execute(action="start", session_id="drain", command="echo one"))
        try:
            _BACKGROUND_PROCESSES["drain"]["thread"].join(timeout=5)  # the reader stops at EOF
            first = asyncio.run(tool.execute(action="output", session_id="drain"))
            second = asyncio.run(tool.execute(action="output", session_id="drain"))
        finally:
            asyncio.run(tool.execute(action="stop", session_id="drain"))
        assert first.output == "one\n"
        assert second.output == ""

    def test_output_over_the_cap_keeps_the_tail_and_spills_the_rest(self) -> None:
        tool = ProcessTool()
        asyncio.run(tool.execute(action="start", session_id="seq", command="seq 1 100000; echo err >&2"))
        try:
            _BACKGROUND_PROCESSES["seq"]["thread"].join(timeout=5)
            first = asyncio.run(tool.execute(action="output", session_id="seq"))
            second = asyncio.run(tool.execute(action="output", session_id="seq"))
        finally:
            asyncio.run(tool.execute(action="stop", session_id="seq"))
        assert _within_the_cap(first.output)
        *tail, blank, notice = first.output.split("\n")
        match = TAIL_NOTICE.fullmatch(notice)
        assert match is not None
        assert blank == ""
        assert tail == [*(str(n) for n in range(100_002 - int(match[1]), 100_001)), "err"]
        # The lines the tail leaves out stay readable in the spill file.
        expected = "".join(f"{n}\n" for n in range(1, 100_001)) + "err\n"
        assert Path(match[5]).read_text(encoding="utf-8") == expected
        assert second.output == ""

    def test_start_rejects_a_session_id_in_use(self) -> None:
        tool = ProcessTool()
        asyncio.run(tool.execute(action="start", session_id="dup", command="exec sleep 30"))
        proc = _BACKGROUND_PROCESSES["dup"]["proc"]
        try:
            again = asyncio.run(tool.execute(action="start", session_id="dup", command="exec sleep 30"))
            assert _BACKGROUND_PROCESSES["dup"]["proc"] is proc
        finally:
            asyncio.run(tool.execute(action="stop", session_id="dup"))
        assert "already exists" in again.error

    def test_stop_kills_the_group(self, tmp_path) -> None:
        tool = ProcessTool()
        command = "sleep 30 & echo $! > pid; wait"
        asyncio.run(tool.execute(action="start", session_id="group", command=command, cwd=str(tmp_path)))
        pid = _read_pid(tmp_path / "pid")
        result = asyncio.run(tool.execute(action="stop", session_id="group"))
        assert result.error is None
        assert "group" not in _BACKGROUND_PROCESSES
        assert not _alive(pid)

    def test_stop_background_processes_terminates_all(self, tmp_path) -> None:
        tool = ProcessTool()
        for name in ("a", "b"):
            command = f"sleep 30 & echo $! > {name}; wait"
            asyncio.run(tool.execute(action="start", session_id=name, command=command, cwd=str(tmp_path)))
        pids = [_read_pid(tmp_path / name) for name in ("a", "b")]
        coding.stop_background_processes()
        assert _BACKGROUND_PROCESSES == {}
        assert not any(_alive(pid) for pid in pids)


# ---- WebFetchTool / BrowserTool: bounded fetches ----


def _response(body: bytes) -> MagicMock:
    """A urlopen response that serves `body`."""
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.side_effect = io.BytesIO(body).read
    response.headers.get_content_charset.return_value = "utf-8"
    return response


FETCH_TOOLS = [
    pytest.param(WebFetchTool, "marvin.tools.coding.safe_urlopen", id="web_fetch"),
    pytest.param(BrowserTool, "marvin.tools.builtin.safe_urlopen", id="browser_fetch"),
]


@pytest.mark.parametrize(("tool_cls", "urlopen"), FETCH_TOOLS)
class TestFetchTools:
    def test_reads_a_bounded_body(self, tool_cls, urlopen) -> None:
        response = _response(b"hello")
        with patch(urlopen, return_value=response):
            result = asyncio.run(tool_cls().execute(url="http://example.com"))
        response.read.assert_called_once_with(FETCH_MAX_BYTES + 1)
        assert result.output == "hello"

    def test_a_long_page_shows_its_head_and_spills_the_text(self, tool_cls, urlopen) -> None:
        body = "".join(f"{n}\n" for n in range(1, 100_001)).encode()
        with patch(urlopen, return_value=_response(body)):
            result = asyncio.run(tool_cls().execute(url="http://example.com"))
        assert _within_the_cap(result.to_model_content())
        *head, blank, notice = result.output.split("\n")
        match = HEAD_NOTICE.fullmatch(notice)
        assert match is not None
        assert blank == ""
        assert head == [str(n) for n in range(1, int(match[1]) + 1)]
        spilled = Path(match[5]).read_text(encoding="utf-8")
        assert spilled.startswith("1\n2\n3\n")
        assert spilled.rstrip("\n").endswith("\n100000")

    def test_a_body_over_the_bound_says_so(self, tool_cls, urlopen, monkeypatch) -> None:
        monkeypatch.setattr(truncate, "FETCH_MAX_BYTES", 1000)
        response = _response(b"x" * 5000)
        with patch(urlopen, return_value=response):
            result = asyncio.run(tool_cls().execute(url="http://example.com"))
        response.read.assert_called_once_with(1001)
        assert result.output == "x" * 1000 + "\n\n[Only the first 1000 bytes of the response body were read.]"


# ---- BrowserAutomationTool ----


def _playwright(page: AsyncMock) -> MagicMock:
    """An `async_playwright()` session whose browser opens `page`."""
    browser = AsyncMock()
    browser.new_page.return_value = page
    playwright = MagicMock()
    playwright.chromium.launch = AsyncMock(return_value=browser)
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=playwright)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


class TestBrowserAutomationTool:
    def test_screenshot_defaults_to_the_system_temp_dir(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        page = AsyncMock()
        with patch("marvin.tools.coding.async_playwright", return_value=_playwright(page)):
            result = asyncio.run(BrowserAutomationTool().execute(action="screenshot"))
        expected = str(tmp_path / "screenshot.png")
        page.screenshot.assert_awaited_once_with(path=expected, full_page=True)
        assert result.data == {"path": expected}

    def test_content_is_capped_with_a_spill_file(self) -> None:
        page = AsyncMock()
        page.content.return_value = "".join(f"<p>{n}</p>\n" for n in range(1, 100_001))
        with patch("marvin.tools.coding.async_playwright", return_value=_playwright(page)):
            result = asyncio.run(BrowserAutomationTool().execute(action="content"))
        assert _within_the_cap(result.output)
        *head, blank, notice = result.output.split("\n")
        match = HEAD_NOTICE.fullmatch(notice)
        assert match is not None
        assert head == [str(n) for n in range(1, int(match[1]) + 1)]
        plain = "\n".join(str(n) for n in range(1, 100_001))
        assert Path(match[5]).read_text(encoding="utf-8") == plain
        assert result.data == {"length": len(plain)}
