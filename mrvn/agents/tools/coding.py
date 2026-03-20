"""Coding tools for file operations, shell execution, web fetching, and agent sessions."""

import asyncio
import base64
import logging
import os
import re
import subprocess
import threading
from typing import Any
from urllib.request import Request, urlopen

from agents.tools.base import BaseTool, ToolParameter, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

try:
    from playwright.async_api import async_playwright  # noqa: PLC0415
except ImportError:
    async_playwright = None  # type: ignore[assignment]



class ReadTool(BaseTool):
    """Read the contents of a file."""

    name = "read"
    description = "Read the contents of a file at the given path. Returns the file content as text."
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Absolute or relative path to the file to read.",
            required=True,
        ),
        ToolParameter(
            name="offset",
            type="number",
            description="Line number to start reading from (1-indexed). Defaults to 1.",
            required=False,
            default=1,
        ),
        ToolParameter(
            name="limit",
            type="number",
            description="Maximum number of lines to read. Defaults to all lines.",
            required=False,
            default=0,
        ),
    ]

    async def execute(self, path: str, offset: int = 1, limit: int = 0) -> ToolResult:
        """Read file contents."""
        try:
            expanded = os.path.expanduser(path)
            with open(expanded, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            start = max(0, offset - 1)
            end = start + limit if limit > 0 else len(lines)
            selected = lines[start:end]

            content = "".join(selected)
            return ToolResult.success(
                output=content,
                data={"path": path, "lines_read": len(selected), "total_lines": len(lines)},
            )
        except FileNotFoundError:
            return ToolResult.from_error(f"File not found: {path}")
        except PermissionError:
            return ToolResult.from_error(f"Permission denied: {path}")
        except OSError as exc:
            return ToolResult.from_error(f"Error reading file: {exc}")


class WriteTool(BaseTool):
    """Write or overwrite a file."""

    name = "write"
    description = "Write content to a file, creating it if it does not exist or overwriting it if it does."
    require_approval = True
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Absolute or relative path to the file to write.",
            required=True,
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Content to write to the file.",
            required=True,
        ),
    ]

    async def execute(self, path: str, content: str) -> ToolResult:
        """Write content to a file."""
        try:
            expanded = os.path.expanduser(path)
            os.makedirs(os.path.dirname(os.path.abspath(expanded)), exist_ok=True)
            with open(expanded, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult.success(
                output=f"Successfully wrote {len(content)} characters to {path}.",
                data={"path": path, "bytes_written": len(content.encode("utf-8"))},
            )
        except PermissionError:
            return ToolResult.from_error(f"Permission denied: {path}")
        except OSError as exc:
            return ToolResult.from_error(f"Error writing file: {exc}")


class EditTool(BaseTool):
    """Perform a targeted string replacement in a file."""

    name = "edit"
    description = (
        "Replace an exact string in a file with new text. "
        "The old_string must appear exactly once in the file."
    )
    require_approval = True
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to edit.",
            required=True,
        ),
        ToolParameter(
            name="old_string",
            type="string",
            description="The exact string to find and replace. Must be unique in the file.",
            required=True,
        ),
        ToolParameter(
            name="new_string",
            type="string",
            description="The replacement string.",
            required=True,
        ),
    ]

    async def execute(self, path: str, old_string: str, new_string: str) -> ToolResult:
        """Replace a string in a file."""
        try:
            expanded = os.path.expanduser(path)
            with open(expanded, encoding="utf-8") as f:
                content = f.read()

            count = content.count(old_string)
            if count == 0:
                return ToolResult.from_error(f"String not found in {path}.")
            if count > 1:
                return ToolResult.from_error(
                    f"String appears {count} times in {path}. Provide more context to make it unique."
                )

            new_content = content.replace(old_string, new_string, 1)
            with open(expanded, "w", encoding="utf-8") as f:
                f.write(new_content)

            return ToolResult.success(
                output=f"Successfully replaced 1 occurrence in {path}.",
                data={"path": path, "replacements": 1},
            )
        except FileNotFoundError:
            return ToolResult.from_error(f"File not found: {path}")
        except PermissionError:
            return ToolResult.from_error(f"Permission denied: {path}")
        except OSError as exc:
            return ToolResult.from_error(f"Error editing file: {exc}")


class ApplyPatchTool(BaseTool):
    """Apply a unified diff patch to one or more files."""

    name = "apply_patch"
    description = (
        "Apply a unified diff patch (as produced by `diff -u` or `git diff`) to files. "
        "Provide the full patch text including file headers."
    )
    require_approval = True
    parameters = [
        ToolParameter(
            name="patch",
            type="string",
            description="The unified diff patch text to apply.",
            required=True,
        ),
    ]

    async def execute(self, patch: str) -> ToolResult:
        """Apply a patch using the `patch` command."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "patch",
                "-p1",
                "--forward",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(input=patch.encode()), timeout=30)
            if proc.returncode == 0:
                return ToolResult.success(
                    output=stdout.decode(),
                    data={"applied": True, "return_code": proc.returncode},
                )
            return ToolResult.from_error(
                f"patch exited with code {proc.returncode}: {stderr.decode()}",
                output=stdout.decode(),
            )
        except FileNotFoundError:
            return ToolResult.from_error("'patch' command not found. Install it with: apt install patch")
        except asyncio.TimeoutError:
            return ToolResult.from_error("patch command timed out")
        except OSError as exc:
            return ToolResult.from_error(f"Error applying patch: {exc}")


class ExecTool(BaseTool):
    """Run a shell command and return output."""

    name = "exec"
    description = (
        "Execute a shell command and return its output. "
        "Provides access to the `gh` CLI for GitHub operations, "
        "git, and any other installed command-line tools."
    )
    require_approval = True
    parameters = [
        ToolParameter(
            name="command",
            type="string",
            description="The shell command to execute.",
            required=True,
        ),
        ToolParameter(
            name="cwd",
            type="string",
            description="Working directory for the command. Defaults to current directory.",
            required=False,
            default="",
        ),
        ToolParameter(
            name="timeout",
            type="number",
            description="Timeout in seconds. Defaults to 60.",
            required=False,
            default=60,
        ),
    ]

    async def execute(self, command: str, cwd: str = "", timeout: int = 60) -> ToolResult:
        """Execute a shell command."""
        work_dir = os.path.expanduser(cwd) if cwd else None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            combined = stdout.decode(errors="replace")
            err_text = stderr.decode(errors="replace")
            output = combined + (f"\n[stderr]\n{err_text}" if err_text else "")
            if proc.returncode == 0:
                return ToolResult.success(
                    output=output,
                    data={"return_code": proc.returncode, "stdout": combined, "stderr": err_text},
                )
            return ToolResult(
                status=ToolStatus.ERROR,
                output=output,
                error=f"Command exited with code {proc.returncode}.",
                data={"return_code": proc.returncode, "stdout": combined, "stderr": err_text},
            )
        except asyncio.TimeoutError:
            return ToolResult.from_error(f"Command timed out after {timeout}s.")
        except OSError as exc:
            return ToolResult.from_error(f"Error executing command: {exc}")


# Background process registry: session_id -> (process, stdout_buffer, stderr_buffer, thread)
_BACKGROUND_PROCESSES: dict[str, dict[str, Any]] = {}
_BACKGROUND_LOCK = threading.Lock()


class ProcessTool(BaseTool):
    """Manage background shell sessions."""

    name = "process"
    description = (
        "Manage long-running background processes. "
        "Actions: start (launch a background command), status (check if running), "
        "output (read buffered output), stop (terminate), list (list all sessions)."
    )
    require_approval = True
    parameters = [
        ToolParameter(
            name="action",
            type="string",
            description="Action: 'start', 'status', 'output', 'stop', 'list'.",
            required=True,
            enum=["start", "status", "output", "stop", "list"],
        ),
        ToolParameter(
            name="session_id",
            type="string",
            description="Session identifier (required for status, output, stop).",
            required=False,
            default="",
        ),
        ToolParameter(
            name="command",
            type="string",
            description="Shell command to run (required for start).",
            required=False,
            default="",
        ),
        ToolParameter(
            name="cwd",
            type="string",
            description="Working directory (used with start).",
            required=False,
            default="",
        ),
    ]

    async def execute(
        self,
        action: str,
        session_id: str = "",
        command: str = "",
        cwd: str = "",
    ) -> ToolResult:
        """Manage background processes."""
        if action == "list":
            with _BACKGROUND_LOCK:
                sessions = list(_BACKGROUND_PROCESSES.keys())
            return ToolResult.success(
                output=f"Active sessions: {sessions}",
                data={"sessions": sessions},
            )

        if action == "start":
            if not command:
                return ToolResult.from_error("command is required for action=start")
            if not session_id:
                return ToolResult.from_error("session_id is required for action=start")
            work_dir = os.path.expanduser(cwd) if cwd else None
            try:
                proc = subprocess.Popen(
                    command,
                    shell=True,  # noqa: S602
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=work_dir,
                    text=True,
                )
                buf: list[str] = []

                def _reader(p: subprocess.Popen, b: list[str]) -> None:
                    for line in p.stdout:
                        b.append(line)

                t = threading.Thread(target=_reader, args=(proc, buf), daemon=True)
                t.start()
                with _BACKGROUND_LOCK:
                    _BACKGROUND_PROCESSES[session_id] = {"proc": proc, "buf": buf, "thread": t}
                return ToolResult.success(
                    output=f"Started session '{session_id}' (pid {proc.pid}).",
                    data={"session_id": session_id, "pid": proc.pid},
                )
            except OSError as exc:
                return ToolResult.from_error(f"Failed to start process: {exc}")

        with _BACKGROUND_LOCK:
            entry = _BACKGROUND_PROCESSES.get(session_id)
        if entry is None:
            return ToolResult.from_error(f"Session '{session_id}' not found.")

        proc = entry["proc"]
        buf = entry["buf"]

        if action == "status":
            running = proc.poll() is None
            return ToolResult.success(
                output=f"Session '{session_id}' is {'running' if running else 'finished'}.",
                data={"session_id": session_id, "running": running, "pid": proc.pid},
            )

        if action == "output":
            with _BACKGROUND_LOCK:
                lines = list(buf)
            return ToolResult.success(
                output="".join(lines),
                data={"session_id": session_id, "lines": len(lines)},
            )

        if action == "stop":
            proc.terminate()
            with _BACKGROUND_LOCK:
                _BACKGROUND_PROCESSES.pop(session_id, None)
            return ToolResult.success(
                output=f"Session '{session_id}' terminated.",
                data={"session_id": session_id},
            )

        return ToolResult.from_error(f"Unknown action: {action}")


class WebFetchTool(BaseTool):
    """Fetch the content of a URL as plain text or markdown."""

    name = "web_fetch"
    description = "Fetch the content of a URL and return it as plain text."
    parameters = [
        ToolParameter(
            name="url",
            type="string",
            description="The URL to fetch.",
            required=True,
        ),
        ToolParameter(
            name="timeout",
            type="number",
            description="Request timeout in seconds. Defaults to 15.",
            required=False,
            default=15,
        ),
    ]

    async def execute(self, url: str, timeout: int = 15) -> ToolResult:
        """Fetch URL content."""
        try:
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, self._fetch, url, timeout)
            return ToolResult.success(
                output=content,
                data={"url": url, "length": len(content)},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.from_error(f"Failed to fetch {url}: {exc}")

    def _fetch(self, url: str, timeout: int) -> str:
        req = Request(url, headers={"User-Agent": "marvin-manager/1.0"})  # noqa: S310
        with urlopen(req, timeout=timeout) as response:  # noqa: S310
            raw = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            text = raw.decode(charset, errors="replace")
        # Strip HTML tags for readability
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class RealWebSearchTool(BaseTool):
    """Search the web using Brave Search API."""

    name = "web_search"
    description = (
        "Search the web using Brave Search API. "
        "Set BRAVE_SEARCH_API_KEY environment variable to enable."
    )
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
        """Search the web."""
        api_key = os.getenv("BRAVE_SEARCH_API_KEY", "")
        if not api_key:
            return ToolResult.from_error(
                "BRAVE_SEARCH_API_KEY environment variable is not set. "
                "Obtain a key from https://api.search.brave.com/ and set it."
            )

        import json  # noqa: PLC0415

        count = min(max(1, num_results), 10)
        url = f"https://api.search.brave.com/res/v1/web/search?q={query}&count={count}"
        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, self._search, url, api_key)
            results = data.get("web", {}).get("results", [])
            formatted = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "description": r.get("description", ""),
                }
                for r in results
            ]
            lines = [f"Search results for '{query}':"]
            for i, r in enumerate(formatted, 1):
                lines.append(f"\n{i}. {r['title']}\n   {r['url']}\n   {r['description']}")
            return ToolResult.success(
                output="\n".join(lines),
                data={"query": query, "results": formatted, "count": len(formatted)},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.from_error(f"Search failed: {exc}")

    def _search(self, url: str, api_key: str) -> dict:
        import json  # noqa: PLC0415

        req = Request(url, headers={"Accept": "application/json", "X-Subscription-Token": api_key})  # noqa: S310
        with urlopen(req, timeout=10) as resp:  # noqa: S310
            return json.loads(resp.read())


# Sub-agent session registry: session_id -> subprocess.Popen
_AGENT_SESSIONS: dict[str, subprocess.Popen] = {}
_AGENT_SESSIONS_LOCK = threading.Lock()


class SessionsSpawnTool(BaseTool):
    """Spawn a sub-agent Claude CLI session."""

    name = "sessions_spawn"
    description = (
        "Spawn a new Claude CLI sub-agent session with an initial prompt. "
        "Returns a session_id that can be used with sessions_send."
    )
    require_approval = True
    parameters = [
        ToolParameter(
            name="session_id",
            type="string",
            description="A unique identifier for this session.",
            required=True,
        ),
        ToolParameter(
            name="prompt",
            type="string",
            description="Initial prompt or system message for the sub-agent.",
            required=True,
        ),
        ToolParameter(
            name="model",
            type="string",
            description="Claude model to use (e.g., 'claude-sonnet-4-6'). Defaults to claude-sonnet-4-6.",
            required=False,
            default="claude-sonnet-4-6",
        ),
    ]

    async def execute(self, session_id: str, prompt: str, model: str = "claude-sonnet-4-6") -> ToolResult:
        """Spawn a Claude CLI sub-agent."""
        with _AGENT_SESSIONS_LOCK:
            if session_id in _AGENT_SESSIONS:
                return ToolResult.from_error(f"Session '{session_id}' already exists.")
        try:
            proc = subprocess.Popen(  # noqa: S603
                ["claude", "--model", model, "--print"],  # noqa: S607
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate(input=prompt, timeout=120)
            if proc.returncode != 0:
                return ToolResult.from_error(
                    f"Sub-agent failed (code {proc.returncode}): {stderr}",
                    output=stdout,
                )
            with _AGENT_SESSIONS_LOCK:
                _AGENT_SESSIONS[session_id] = proc
            return ToolResult.success(
                output=stdout,
                data={"session_id": session_id, "model": model},
            )
        except FileNotFoundError:
            return ToolResult.from_error("'claude' CLI not found. Install the Claude CLI to use sessions.")
        except subprocess.TimeoutExpired:
            return ToolResult.from_error("Sub-agent timed out.")
        except OSError as exc:
            return ToolResult.from_error(f"Failed to spawn session: {exc}")


class SessionsSendTool(BaseTool):
    """Send a message to a spawned sub-agent session."""

    name = "sessions_send"
    description = "Send a prompt to an existing Claude CLI sub-agent session and return the response."
    require_approval = True
    parameters = [
        ToolParameter(
            name="session_id",
            type="string",
            description="The session identifier (from sessions_spawn).",
            required=True,
        ),
        ToolParameter(
            name="prompt",
            type="string",
            description="The message/prompt to send.",
            required=True,
        ),
        ToolParameter(
            name="model",
            type="string",
            description="Claude model to use. Defaults to claude-sonnet-4-6.",
            required=False,
            default="claude-sonnet-4-6",
        ),
    ]

    async def execute(self, session_id: str, prompt: str, model: str = "claude-sonnet-4-6") -> ToolResult:
        """Send a message to a sub-agent via Claude CLI."""
        try:
            proc = subprocess.Popen(  # noqa: S603
                ["claude", "--model", model, "--print"],  # noqa: S607
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate(input=prompt, timeout=120)
            if proc.returncode != 0:
                return ToolResult.from_error(
                    f"Sub-agent failed (code {proc.returncode}): {stderr}",
                    output=stdout,
                )
            return ToolResult.success(
                output=stdout,
                data={"session_id": session_id},
            )
        except FileNotFoundError:
            return ToolResult.from_error("'claude' CLI not found.")
        except subprocess.TimeoutExpired:
            return ToolResult.from_error("Sub-agent timed out.")
        except OSError as exc:
            return ToolResult.from_error(f"Failed to send message: {exc}")


class ImageTool(BaseTool):
    """Analyze an image using a vision-capable LLM."""

    name = "image"
    description = (
        "Analyze an image file or URL using an LLM with vision capability. "
        "Returns a description or answer to a prompt about the image."
    )
    parameters = [
        ToolParameter(
            name="source",
            type="string",
            description="Path to an image file or a URL of the image.",
            required=True,
        ),
        ToolParameter(
            name="prompt",
            type="string",
            description="Question or instruction about the image. Defaults to 'Describe this image.'",
            required=False,
            default="Describe this image.",
        ),
    ]

    async def execute(self, source: str, prompt: str = "Describe this image.") -> ToolResult:
        """Analyze an image."""
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return ToolResult.from_error("ANTHROPIC_API_KEY environment variable is not set.")
        try:
            image_data, media_type = await self._load_image(source)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._call_vision,
                api_key,
                image_data,
                media_type,
                prompt,
            )
            return ToolResult.success(
                output=result,
                data={"source": source, "prompt": prompt},
            )
        except ImportError:
            return ToolResult.from_error("anthropic package is required. Install it with: pip install anthropic")
        except Exception as exc:  # noqa: BLE001
            return ToolResult.from_error(f"Image analysis failed: {exc}")

    async def _load_image(self, source: str) -> tuple[str, str]:
        """Load image as base64 and detect media type."""
        if source.startswith(("http://", "https://")):
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(None, self._fetch_url, source)
        else:
            with open(os.path.expanduser(source), "rb") as f:
                raw = f.read()

        encoded = base64.b64encode(raw).decode()
        # Detect media type from magic bytes
        if raw[:4] == b"\x89PNG":
            media_type = "image/png"
        elif raw[:2] == b"\xff\xd8":
            media_type = "image/jpeg"
        elif raw[:4] == b"GIF8":
            media_type = "image/gif"
        elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            media_type = "image/webp"
        else:
            media_type = "image/jpeg"
        return encoded, media_type

    def _fetch_url(self, url: str) -> bytes:
        req = Request(url, headers={"User-Agent": "marvin-manager/1.0"})  # noqa: S310
        with urlopen(req, timeout=30) as resp:  # noqa: S310
            return resp.read()

    def _call_vision(self, api_key: str, image_data: str, media_type: str, prompt: str) -> str:
        import anthropic  # noqa: PLC0415

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return message.content[0].text


class BrowserTool(BaseTool):
    """Browser automation using Playwright."""

    name = "browser"
    description = (
        "Control a web browser to navigate pages, click elements, fill forms, and extract content. "
        "Requires the 'playwright' package and browser binaries."
    )
    require_approval = True
    parameters = [
        ToolParameter(
            name="action",
            type="string",
            description="Action: 'navigate', 'click', 'type', 'screenshot', 'content', 'evaluate'.",
            required=True,
            enum=["navigate", "click", "type", "screenshot", "content", "evaluate"],
        ),
        ToolParameter(
            name="url",
            type="string",
            description="URL to navigate to (required for action=navigate).",
            required=False,
            default="",
        ),
        ToolParameter(
            name="selector",
            type="string",
            description="CSS selector for the target element (used with click, type).",
            required=False,
            default="",
        ),
        ToolParameter(
            name="text",
            type="string",
            description="Text to type (used with action=type).",
            required=False,
            default="",
        ),
        ToolParameter(
            name="script",
            type="string",
            description="JavaScript to evaluate in the page (used with action=evaluate).",
            required=False,
            default="",
        ),
        ToolParameter(
            name="output_path",
            type="string",
            description="Path to save screenshot (used with action=screenshot).",
            required=False,
            default="/tmp/screenshot.png",
        ),
    ]

    async def execute(
        self,
        action: str,
        url: str = "",
        selector: str = "",
        text: str = "",
        script: str = "",
        output_path: str = "/tmp/screenshot.png",
    ) -> ToolResult:
        """Perform browser action."""
        if async_playwright is None:
            return ToolResult.from_error(
                "playwright package is required. Install it with: pip install playwright && playwright install chromium"
            )

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                page = await browser.new_page()

                if action == "navigate":
                    if not url:
                        return ToolResult.from_error("url is required for action=navigate")
                    await page.goto(url, timeout=30000)
                    title = await page.title()
                    await browser.close()
                    return ToolResult.success(
                        output=f"Navigated to {url} — title: {title}",
                        data={"url": url, "title": title},
                    )

                if action == "content":
                    if url:
                        await page.goto(url, timeout=30000)
                    content = await page.content()
                    # Strip tags
                    plain = re.sub(r"<[^>]+>", "", content)
                    plain = re.sub(r"\n{3,}", "\n\n", plain).strip()
                    await browser.close()
                    return ToolResult.success(
                        output=plain[:10000],
                        data={"length": len(plain)},
                    )

                if action == "screenshot":
                    if url:
                        await page.goto(url, timeout=30000)
                    await page.screenshot(path=output_path, full_page=True)
                    await browser.close()
                    return ToolResult.success(
                        output=f"Screenshot saved to {output_path}.",
                        data={"path": output_path},
                    )

                if action == "click":
                    if not selector:
                        return ToolResult.from_error("selector is required for action=click")
                    if url:
                        await page.goto(url, timeout=30000)
                    await page.click(selector, timeout=10000)
                    await browser.close()
                    return ToolResult.success(
                        output=f"Clicked element: {selector}",
                        data={"selector": selector},
                    )

                if action == "type":
                    if not selector:
                        return ToolResult.from_error("selector is required for action=type")
                    if url:
                        await page.goto(url, timeout=30000)
                    await page.fill(selector, text)
                    await browser.close()
                    return ToolResult.success(
                        output=f"Typed into {selector}.",
                        data={"selector": selector, "text": text},
                    )

                if action == "evaluate":
                    if not script:
                        return ToolResult.from_error("script is required for action=evaluate")
                    if url:
                        await page.goto(url, timeout=30000)
                    result = await page.evaluate(script)
                    await browser.close()
                    return ToolResult.success(
                        output=str(result),
                        data={"result": result},
                    )

                await browser.close()
                return ToolResult.from_error(f"Unknown action: {action}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult.from_error(f"Browser error: {exc}")


def register_coding_tools(registry: Any) -> None:
    """Register all coding tools with the given registry.

    Args:
        registry: The ToolRegistry instance.
    """
    tools = [
        ReadTool(),
        WriteTool(),
        EditTool(),
        ApplyPatchTool(),
        ExecTool(),
        ProcessTool(),
        WebFetchTool(),
        RealWebSearchTool(),
        SessionsSpawnTool(),
        SessionsSendTool(),
        ImageTool(),
        BrowserTool(),
    ]
    for tool in tools:
        try:
            registry.register(tool)
        except ValueError:
            logger.debug("Tool %s already registered", tool.name)
