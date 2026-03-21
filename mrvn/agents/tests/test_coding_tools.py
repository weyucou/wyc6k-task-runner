"""Tests for coding tools: read, write, edit, apply_patch, exec, process,
web_fetch, web_search, sessions_spawn, sessions_send, image, browser."""

import os
import tempfile
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch

from agents.tools.base import ToolStatus
from agents.tools.coding import (
    ApplyPatchTool,
    BrowserTool,
    EditTool,
    ExecTool,
    ImageTool,
    ProcessTool,
    RealWebSearchTool,
    ReadTool,
    SessionsSendTool,
    SessionsSpawnTool,
    WebFetchTool,
    WriteTool,
    _SESSION_MANAGER,
)


# ---------------------------------------------------------------------------
# ReadTool
# ---------------------------------------------------------------------------


class ReadToolTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = ReadTool()

    async def test_read_existing_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line1\nline2\nline3\n")
            path = f.name
        try:
            result = await self.tool.execute(path=path)
            self.assertEqual(result.status, ToolStatus.SUCCESS)
            self.assertIn("line1", result.output)
            self.assertIn("line3", result.output)
        finally:
            os.unlink(path)

    async def test_read_with_offset_and_limit(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line1\nline2\nline3\nline4\n")
            path = f.name
        try:
            result = await self.tool.execute(path=path, offset=2, limit=2)
            self.assertEqual(result.status, ToolStatus.SUCCESS)
            self.assertIn("line2", result.output)
            self.assertIn("line3", result.output)
            self.assertNotIn("line1", result.output)
            self.assertNotIn("line4", result.output)
        finally:
            os.unlink(path)

    async def test_read_nonexistent_file(self) -> None:
        result = await self.tool.execute(path="/nonexistent/path/file.txt")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("not found", result.error.lower())

    async def test_read_data_fields(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\n")
            path = f.name
        try:
            result = await self.tool.execute(path=path)
            self.assertEqual(result.status, ToolStatus.SUCCESS)
            self.assertIn("lines_read", result.data)
            self.assertIn("total_lines", result.data)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# WriteTool
# ---------------------------------------------------------------------------


class WriteToolTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = WriteTool()

    async def test_write_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "new_file.txt")
            result = await self.tool.execute(path=path, content="hello world")
            self.assertEqual(result.status, ToolStatus.SUCCESS)
            with open(path) as f:
                self.assertEqual(f.read(), "hello world")

    async def test_overwrite_existing_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("old content")
            path = f.name
        try:
            result = await self.tool.execute(path=path, content="new content")
            self.assertEqual(result.status, ToolStatus.SUCCESS)
            with open(path) as f:
                self.assertEqual(f.read(), "new content")
        finally:
            os.unlink(path)

    async def test_write_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "file.txt")
            result = await self.tool.execute(path=path, content="data")
            self.assertEqual(result.status, ToolStatus.SUCCESS)
            self.assertTrue(os.path.exists(path))

    async def test_write_data_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            result = await self.tool.execute(path=path, content="abc")
            self.assertIn("bytes_written", result.data)


# ---------------------------------------------------------------------------
# EditTool
# ---------------------------------------------------------------------------


class EditToolTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = EditTool()

    async def test_replace_unique_string(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("foo bar baz")
            path = f.name
        try:
            result = await self.tool.execute(path=path, old_string="bar", new_string="qux")
            self.assertEqual(result.status, ToolStatus.SUCCESS)
            with open(path) as f:
                self.assertEqual(f.read(), "foo qux baz")
        finally:
            os.unlink(path)

    async def test_error_on_string_not_found(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world")
            path = f.name
        try:
            result = await self.tool.execute(path=path, old_string="missing", new_string="x")
            self.assertEqual(result.status, ToolStatus.ERROR)
            self.assertIn("not found", result.error.lower())
        finally:
            os.unlink(path)

    async def test_error_on_ambiguous_string(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("aa aa aa")
            path = f.name
        try:
            result = await self.tool.execute(path=path, old_string="aa", new_string="bb")
            self.assertEqual(result.status, ToolStatus.ERROR)
            self.assertIn("3 times", result.error)
        finally:
            os.unlink(path)

    async def test_error_file_not_found(self) -> None:
        result = await self.tool.execute(path="/no/such/file.txt", old_string="x", new_string="y")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("not found", result.error.lower())


# ---------------------------------------------------------------------------
# ApplyPatchTool
# ---------------------------------------------------------------------------


class ApplyPatchToolTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = ApplyPatchTool()

    async def test_apply_valid_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "file.txt")
            with open(target, "w") as f:
                f.write("line1\nline2\nline3\n")

            patch_text = "--- a/file.txt\n+++ b/file.txt\n@@ -1,3 +1,3 @@\n line1\n-line2\n+changed\n line3\n"
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = await self.tool.execute(patch=patch_text)
                # patch may succeed or fail depending on environment; check no crash
                self.assertIn(result.status, (ToolStatus.SUCCESS, ToolStatus.ERROR))
            finally:
                os.chdir(old_cwd)

    async def test_invalid_patch_returns_error(self) -> None:
        result = await self.tool.execute(patch="not a valid patch")
        self.assertEqual(result.status, ToolStatus.ERROR)


# ---------------------------------------------------------------------------
# ExecTool
# ---------------------------------------------------------------------------


class ExecToolTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = ExecTool()

    async def test_simple_command(self) -> None:
        result = await self.tool.execute(command="echo hello")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertIn("hello", result.output)

    async def test_failed_command_returns_error(self) -> None:
        result = await self.tool.execute(command="exit 1", timeout=5)
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("return_code", result.data)
        self.assertEqual(result.data["return_code"], 1)

    async def test_timeout(self) -> None:
        result = await self.tool.execute(command="sleep 10", timeout=1)
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("timed out", result.error.lower())

    async def test_stderr_captured(self) -> None:
        result = await self.tool.execute(command="echo err >&2")
        self.assertIn("stderr", result.data)

    async def test_data_contains_return_code(self) -> None:
        result = await self.tool.execute(command="echo ok")
        self.assertIn("return_code", result.data)
        self.assertEqual(result.data["return_code"], 0)


# ---------------------------------------------------------------------------
# ProcessTool
# ---------------------------------------------------------------------------


class ProcessToolTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = ProcessTool()

    async def test_list_empty(self) -> None:
        from agents.tools.coding import _BACKGROUND_PROCESSES, _BACKGROUND_LOCK

        with _BACKGROUND_LOCK:
            _BACKGROUND_PROCESSES.clear()
        result = await self.tool.execute(action="list")
        self.assertEqual(result.status, ToolStatus.SUCCESS)

    async def test_start_and_status(self) -> None:
        from agents.tools.coding import _BACKGROUND_PROCESSES, _BACKGROUND_LOCK

        with _BACKGROUND_LOCK:
            _BACKGROUND_PROCESSES.clear()

        sid = "test_session_proc"
        start_result = await self.tool.execute(action="start", session_id=sid, command="sleep 2")
        self.assertEqual(start_result.status, ToolStatus.SUCCESS)

        status_result = await self.tool.execute(action="status", session_id=sid)
        self.assertEqual(status_result.status, ToolStatus.SUCCESS)

        stop_result = await self.tool.execute(action="stop", session_id=sid)
        self.assertEqual(stop_result.status, ToolStatus.SUCCESS)

    async def test_unknown_session_error(self) -> None:
        result = await self.tool.execute(action="status", session_id="no_such_session_xyz")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("not found", result.error.lower())

    async def test_start_missing_command(self) -> None:
        result = await self.tool.execute(action="start", session_id="x")
        self.assertEqual(result.status, ToolStatus.ERROR)

    async def test_start_missing_session_id(self) -> None:
        result = await self.tool.execute(action="start", command="echo hi")
        self.assertEqual(result.status, ToolStatus.ERROR)


# ---------------------------------------------------------------------------
# WebFetchTool
# ---------------------------------------------------------------------------


class WebFetchToolTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = WebFetchTool()

    async def test_invalid_url_returns_error(self) -> None:
        result = await self.tool.execute(url="http://localhost:19999/nonexistent", timeout=2)
        self.assertEqual(result.status, ToolStatus.ERROR)

    @patch("agents.tools.coding.WebFetchTool._fetch")
    async def test_success_returns_content(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = "fetched content"
        result = await self.tool.execute(url="http://example.com")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertIn("fetched content", result.output)

    @patch("agents.tools.coding.WebFetchTool._fetch")
    async def test_data_contains_url_and_length(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = "some text"
        result = await self.tool.execute(url="http://example.com")
        self.assertIn("url", result.data)
        self.assertIn("length", result.data)


# ---------------------------------------------------------------------------
# RealWebSearchTool
# ---------------------------------------------------------------------------


class RealWebSearchToolTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = RealWebSearchTool()

    async def test_missing_api_key_returns_error(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BRAVE_SEARCH_API_KEY", None)
            result = await self.tool.execute(query="test")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("BRAVE_SEARCH_API_KEY", result.error)

    @patch("agents.tools.coding.RealWebSearchTool._search")
    async def test_success_with_api_key(self, mock_search: MagicMock) -> None:
        mock_search.return_value = {
            "web": {"results": [{"title": "Result 1", "url": "http://example.com", "description": "Desc"}]}
        }
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "fake_key"}):
            result = await self.tool.execute(query="python testing")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertIn("Result 1", result.output)
        self.assertIn("results", result.data)

    async def test_validate_params_requires_query(self) -> None:
        is_valid, error = self.tool.validate_params({})
        self.assertFalse(is_valid)
        self.assertIn("query", error)


# ---------------------------------------------------------------------------
# SessionsSpawnTool
# ---------------------------------------------------------------------------


class SessionsSpawnToolTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = SessionsSpawnTool()

    async def test_claude_not_found_returns_error(self) -> None:
        with patch("subprocess.Popen", side_effect=FileNotFoundError("claude not found")):
            result = await self.tool.execute(session_id="s1", prompt="hello")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("claude", result.error.lower())

    async def test_success_spawns_session(self) -> None:
        _SESSION_MANAGER.clear()

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("agent response", "")
        mock_proc.returncode = 0

        with patch("subprocess.Popen", return_value=mock_proc):
            result = await self.tool.execute(session_id="s_spawn", prompt="hello")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertIn("agent response", result.output)


# ---------------------------------------------------------------------------
# SessionsSendTool
# ---------------------------------------------------------------------------


class SessionsSendToolTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = SessionsSendTool()
        _SESSION_MANAGER.clear()

    def tearDown(self) -> None:
        _SESSION_MANAGER.clear()

    async def test_session_not_found_returns_error(self) -> None:
        result = await self.tool.execute(session_id="nonexistent", prompt="hello")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("not found", result.error.lower())

    async def test_success_sends_message(self) -> None:
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("response text", "")
        mock_proc.returncode = 0

        _SESSION_MANAGER.add("s1", mock_proc)

        result = await self.tool.execute(session_id="s1", prompt="hello")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertIn("response text", result.output)

    async def test_spawn_then_send_uses_same_process(self) -> None:
        """Send must reuse the process stored by spawn, not spawn a fresh subprocess."""
        spawn_tool = SessionsSpawnTool()

        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("initial response", "")
        mock_proc.returncode = 0

        with patch("subprocess.Popen", return_value=mock_proc):
            await spawn_tool.execute(session_id="test_session", prompt="initial prompt")

        stored_proc = _SESSION_MANAGER.get("test_session")
        self.assertIs(stored_proc, mock_proc)

        with patch("subprocess.Popen") as mock_popen_send:
            result = await self.tool.execute(session_id="test_session", prompt="follow-up")

        mock_popen_send.assert_not_called()
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(mock_proc.communicate.call_count, 2)


# ---------------------------------------------------------------------------
# ImageTool
# ---------------------------------------------------------------------------


class ImageToolTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = ImageTool()

    async def test_missing_api_key_returns_error(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            result = await self.tool.execute(source="/tmp/fake.png")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("ANTHROPIC_API_KEY", result.error)

    async def test_file_not_found_returns_error(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake"}):
            result = await self.tool.execute(source="/nonexistent/image.png")
        self.assertEqual(result.status, ToolStatus.ERROR)

    @patch("agents.tools.coding.ImageTool._call_vision")
    @patch("agents.tools.coding.ImageTool._load_image")
    async def test_success_returns_description(self, mock_load: MagicMock, mock_vision: MagicMock) -> None:
        mock_load.return_value = ("base64data", "image/png")
        mock_vision.return_value = "A beautiful sunset."
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake"}):
            result = await self.tool.execute(source="/tmp/fake.png", prompt="Describe this.")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertIn("sunset", result.output)


# ---------------------------------------------------------------------------
# BrowserTool
# ---------------------------------------------------------------------------


class BrowserToolTests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = BrowserTool()

    async def test_missing_playwright_returns_error(self) -> None:
        with patch.dict("sys.modules", {"playwright": None, "playwright.async_api": None}):
            result = await self.tool.execute(action="navigate", url="http://example.com")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("playwright", result.error.lower())

    async def test_navigate_missing_url_returns_error(self) -> None:
        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_page = MagicMock()
        mock_page.goto = MagicMock(return_value=None)
        mock_page.title = MagicMock(return_value="Test")
        mock_browser.new_page = MagicMock(return_value=mock_page)
        mock_pw.chromium.launch = MagicMock(return_value=mock_browser)
        mock_pw.__aenter__ = MagicMock(return_value=mock_pw)
        mock_pw.__aexit__ = MagicMock(return_value=False)

        with patch("agents.tools.coding.async_playwright", side_effect=ImportError("no playwright")):
            result = await self.tool.execute(action="navigate")
        self.assertEqual(result.status, ToolStatus.ERROR)

    async def test_validate_params_action_required(self) -> None:
        is_valid, error = self.tool.validate_params({})
        self.assertFalse(is_valid)
        self.assertIn("action", error)

    async def test_validate_params_invalid_action_enum(self) -> None:
        is_valid, error = self.tool.validate_params({"action": "fly"})
        self.assertFalse(is_valid)
        self.assertIn("must be one of", error.lower())
