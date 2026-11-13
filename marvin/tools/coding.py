# Copyright (c) 2026 MNKT
"""Coding tools for file operations, shell execution, web fetching, and agent sessions."""

import asyncio
import base64
import logging
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any
from urllib.request import Request, urlopen

from marvin.tools.base import BaseTool, ToolParameter, ToolResult, ToolStatus
from marvin.tools.urls import check_url, safe_urlopen

if TYPE_CHECKING:
    from collections.abc import Callable

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
            with Path(path).expanduser().open(encoding="utf-8", errors="replace") as f:
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
            expanded = Path(path).expanduser()
            expanded.resolve().parent.mkdir(parents=True, exist_ok=True)
            expanded.write_text(content, encoding="utf-8")
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
        "Replace an exact string in a file with new text. The old_string must appear exactly once in the file."
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
            expanded = Path(path).expanduser()
            content = expanded.read_text(encoding="utf-8")

            count = content.count(old_string)
            if count == 0:
                return ToolResult.from_error(f"String not found in {path}.")
            if count > 1:
                return ToolResult.from_error(
                    f"String appears {count} times in {path}. Provide more context to make it unique."
                )

            new_content = content.replace(old_string, new_string, 1)
            expanded.write_text(new_content, encoding="utf-8")

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
        except TimeoutError:
            return ToolResult.from_error("patch command timed out")
        except OSError as exc:
            return ToolResult.from_error(f"Error applying patch: {exc}")


_EXEC_DEFAULT_TIMEOUT = 120  # seconds
_KILL_GRACE_SECONDS = 3.0  # how long a process group gets to exit after SIGTERM before SIGKILL
_PIPE_DRAIN_SECONDS = 0.5  # how long the output pipes may stay open after the shell exits
_POLL_SECONDS = 0.05
_READ_CHUNK_BYTES = 64 * 1024


def _signal_group(pgid: int, sig: int) -> bool:
    """Send `sig` to process group `pgid`; False once the group has no members left."""
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    return True


async def _poll_until(condition: Callable[[], bool], seconds: float) -> bool:
    """Check `condition` every _POLL_SECONDS until it holds (True) or `seconds` pass (False)."""
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_POLL_SECONDS)
    return True


async def _terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    """SIGTERM the process group `proc` leads, SIGKILL it if any member outlives the grace period, then reap `proc`."""
    if _signal_group(proc.pid, signal.SIGTERM):
        await _poll_until(lambda: not _signal_group(proc.pid, 0), _KILL_GRACE_SECONDS)
        _signal_group(proc.pid, signal.SIGKILL)
    await _poll_until(lambda: proc.returncode is not None, _KILL_GRACE_SECONDS)


async def _read_into(stream: asyncio.StreamReader, sink: bytearray) -> None:
    """Copy `stream` into `sink` until EOF; what was read survives cancellation."""
    while chunk := await stream.read(_READ_CHUNK_BYTES):
        sink.extend(chunk)


class ExecTool(BaseTool):
    """Primary interface for invoking CLI tools and shell commands."""

    name = "exec"
    description = (
        "The primary interface for invoking CLI tools and shell commands. "
        "Use this to call any installed CLI — gh, git, askcc, aws, terraform, or any other tool on PATH. "
        "Prefer this over specialised wrappers when a CLI already exists for the task. "
        "Commands run with stdin closed and are terminated on timeout; "
        "start servers and other background jobs with the `process` tool. "
        "Increase timeout for long-running tools (e.g. askcc spawns a Claude session and may take hours)."
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
            description=(
                f"Timeout in seconds. Defaults to {_EXEC_DEFAULT_TIMEOUT}. Pass a larger value for long-running CLIs."
            ),
            required=False,
            default=_EXEC_DEFAULT_TIMEOUT,
        ),
    ]

    async def execute(self, command: str, cwd: str = "", timeout: int = _EXEC_DEFAULT_TIMEOUT) -> ToolResult:
        """Execute a shell command in its own process group, with stdin closed."""
        work_dir = Path(cwd).expanduser() if cwd else None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                start_new_session=True,  # the shell leads a new process group, so killpg reaches its children
            )
        except OSError as exc:
            return ToolResult.from_error(f"Error executing command: {exc}")

        assert proc.stdout is not None  # both pipes exist: stdout and stderr are PIPE
        assert proc.stderr is not None
        stdout, stderr = bytearray(), bytearray()
        readers = [
            asyncio.create_task(_read_into(proc.stdout, stdout)),
            asyncio.create_task(_read_into(proc.stderr, stderr)),
        ]
        pipes_held = False
        try:
            # Watch the shell itself: Process.wait() also waits for background children that inherited the pipes.
            exited = await _poll_until(lambda: proc.returncode is not None, timeout)
            if exited:
                _, pending = await asyncio.wait(readers, timeout=_PIPE_DRAIN_SECONDS)
                pipes_held = bool(pending)
            if pipes_held or not exited:
                await _terminate_process_group(proc)
                await asyncio.wait(readers, timeout=_PIPE_DRAIN_SECONDS)
        except asyncio.CancelledError:
            await _terminate_process_group(proc)
            raise
        finally:
            for reader in readers:
                reader.cancel()

        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")
        output = out + (f"\n[stderr]\n{err}" if err else "")
        if not exited:
            message = (
                f"Command timed out after {timeout}s and was terminated. "
                "Rerun with a larger timeout (seconds) if it needs more time."
            )
            # The LLM clients forward only `output` to the model.
            return ToolResult.from_error(message, output=f"{output}\n[{message}]")
        if pipes_held:
            output += (
                "\n[Background processes started by the command kept its output open and were terminated. "
                "Use the `process` tool for long-running jobs.]"
            )
        data = {"return_code": proc.returncode, "stdout": out, "stderr": err}
        if proc.returncode == 0:
            return ToolResult.success(output=output, data=data)
        return ToolResult(
            status=ToolStatus.ERROR,
            output=output,
            error=f"Command exited with code {proc.returncode}.",
            data=data,
        )


# Background process registry: session_id -> {"proc": Popen, "buf": output lines not yet read, "thread": reader}
_BACKGROUND_PROCESSES: dict[str, dict[str, Any]] = {}
_BACKGROUND_LOCK = threading.Lock()
_PROCESS_BUFFER_MAX_LINES = 2000  # per session; older unread lines are dropped


def _terminate_process_groups(procs: list[subprocess.Popen]) -> None:
    """SIGTERM each process group, SIGKILL the groups that outlive one shared grace period, then reap."""
    alive = [proc for proc in procs if _signal_group(proc.pid, signal.SIGTERM)]
    deadline = time.monotonic() + _KILL_GRACE_SECONDS
    while alive and time.monotonic() < deadline:
        time.sleep(_POLL_SECONDS)
        # poll() reaps an exited shell; until then its zombie keeps the group alive
        alive = [proc for proc in alive if proc.poll() is None or _signal_group(proc.pid, 0)]
    for proc in alive:
        _signal_group(proc.pid, signal.SIGKILL)
    for proc in procs:
        proc.wait()


def stop_background_processes() -> None:
    """Terminate every `process` session; call when an agent run ends."""
    with _BACKGROUND_LOCK:
        procs = [entry["proc"] for entry in _BACKGROUND_PROCESSES.values()]
        _BACKGROUND_PROCESSES.clear()
    _terminate_process_groups(procs)


class ProcessTool(BaseTool):
    """Manage background shell sessions."""

    name = "process"
    description = (
        "Manage long-running background processes. "
        "Actions: start (launch a background command), status (check if running), "
        f"output (read the lines printed since the last output call; up to {_PROCESS_BUFFER_MAX_LINES} are kept), "
        "stop (terminate), list (list all sessions). Sessions are stopped when the run ends."
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
            with _BACKGROUND_LOCK:
                if session_id in _BACKGROUND_PROCESSES:
                    return ToolResult.from_error(
                        f"Session '{session_id}' already exists. Stop it first or use another session_id."
                    )
            work_dir = Path(cwd).expanduser() if cwd else None
            try:
                proc = subprocess.Popen(  # noqa: S602
                    command,
                    shell=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=work_dir,
                    text=True,
                    start_new_session=True,  # its own process group, so stop reaches the shell's children
                )
                buf: deque[str] = deque(maxlen=_PROCESS_BUFFER_MAX_LINES)

                def _reader(stream: IO[str], b: deque[str]) -> None:
                    with stream:  # closed at EOF, so a long-lived worker does not leak one pipe per session
                        for line in stream:  # line by line, so `output` sees lines as they arrive
                            with _BACKGROUND_LOCK:
                                b.append(line)

                t = threading.Thread(target=_reader, args=(proc.stdout, buf), daemon=True)
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
                buf.clear()
            return ToolResult.success(
                output="".join(lines),
                data={"session_id": session_id, "lines": len(lines)},
            )

        if action == "stop":
            with _BACKGROUND_LOCK:
                _BACKGROUND_PROCESSES.pop(session_id, None)
            await asyncio.to_thread(_terminate_process_groups, [proc])
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
        with safe_urlopen(req, timeout=timeout) as response:
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
    description = "Search the web using Brave Search API. Set BRAVE_SEARCH_API_KEY environment variable to enable."
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


class AgentSessionManager:
    """Thread-safe registry of spawned sub-agent sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def __contains__(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def get(self, session_id: str) -> subprocess.Popen | None:
        with self._lock:
            return self._sessions.get(session_id)

    def add(self, session_id: str, proc: subprocess.Popen) -> None:
        with self._lock:
            self._sessions[session_id] = proc

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


_SESSION_MANAGER = AgentSessionManager()


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
        if session_id in _SESSION_MANAGER:
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
            _SESSION_MANAGER.add(session_id, proc)
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
        proc = _SESSION_MANAGER.get(session_id)
        if proc is None:
            return ToolResult.from_error(f"Session '{session_id}' not found. Use sessions_spawn first.")
        try:
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
            raw = Path(source).expanduser().read_bytes()

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
        with safe_urlopen(req, timeout=30) as resp:
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
        return "".join(block.text for block in message.content if block.type == "text")


class BrowserAutomationTool(BaseTool):
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
            description=(
                "Path to save screenshot (used with action=screenshot). "
                "Defaults to screenshot.png in the system temp directory."
            ),
            required=False,
            default="",
        ),
    ]

    async def execute(
        self,
        *,
        action: str,
        url: str = "",
        selector: str = "",
        text: str = "",
        script: str = "",
        output_path: str = "",
    ) -> ToolResult:
        """Perform browser action."""
        if async_playwright is None:
            return ToolResult.from_error(
                "playwright package is required. Install it with: pip install playwright && playwright install chromium"
            )

        try:
            if url:
                # Covers every page.goto below. Redirects and subresources inside Chromium are not checked.
                await asyncio.to_thread(check_url, url)
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
                    output_path = output_path or str(Path(tempfile.gettempdir()) / "screenshot.png")
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


class GitHubIssueTool(BaseTool):
    """Interact with GitHub issues via the gh CLI."""

    name = "github_issue"
    description = (
        "View, comment on, label, or close a GitHub issue using the gh CLI. Requires GITHUB_TOKEN environment variable."
    )
    require_approval = False
    parameters = [
        ToolParameter(
            name="action",
            type="string",
            description="Action: 'view', 'comment', 'add_label', 'remove_label', 'close'.",
            required=True,
            enum=["view", "comment", "add_label", "remove_label", "close"],
        ),
        ToolParameter(
            name="issue_url",
            type="string",
            description="Full URL of the GitHub issue (e.g. https://github.com/owner/repo/issues/1).",
            required=True,
        ),
        ToolParameter(
            name="body",
            type="string",
            description="Comment body (required for action=comment).",
            required=False,
            default="",
        ),
        ToolParameter(
            name="label",
            type="string",
            description="Label name (required for action=add_label or action=remove_label).",
            required=False,
            default="",
        ),
    ]

    async def execute(
        self,
        action: str,
        issue_url: str,
        body: str = "",
        label: str = "",
    ) -> ToolResult:
        """Execute a GitHub issue action via gh CLI."""
        token = os.getenv("GITHUB_TOKEN", "")
        if not token:
            return ToolResult.from_error("GITHUB_TOKEN environment variable is not set")

        env = {**os.environ, "GITHUB_TOKEN": token}

        if action == "view":
            cmd = ["gh", "issue", "view", issue_url]
        elif action == "comment":
            if not body:
                return ToolResult.from_error("body is required for action=comment")
            cmd = ["gh", "issue", "comment", issue_url, "--body", body]
        elif action == "add_label":
            if not label:
                return ToolResult.from_error("label is required for action=add_label")
            cmd = ["gh", "issue", "edit", issue_url, "--add-label", label]
        elif action == "remove_label":
            if not label:
                return ToolResult.from_error("label is required for action=remove_label")
            cmd = ["gh", "issue", "edit", issue_url, "--remove-label", label]
        elif action == "close":
            cmd = ["gh", "issue", "close", issue_url]
        else:
            return ToolResult.from_error(f"Unknown action: {action}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            out = stdout.decode(errors="replace")
            err = stderr.decode(errors="replace")
            if proc.returncode == 0:
                return ToolResult.success(
                    output=out or f"Action '{action}' completed successfully.",
                    data={"action": action, "issue_url": issue_url, "return_code": proc.returncode},
                )
            return ToolResult.from_error(
                f"gh exited with code {proc.returncode}: {err}",
                output=out,
            )
        except FileNotFoundError:
            return ToolResult.from_error("'gh' CLI not found. Install the GitHub CLI to use this tool.")
        except TimeoutError:
            return ToolResult.from_error("gh command timed out")
        except OSError as exc:
            return ToolResult.from_error(f"Error running gh: {exc}")


class GitHubPRTool(BaseTool):
    """Interact with GitHub pull requests via the gh CLI."""

    name = "github_pr"
    description = (
        "Create, view, or update a GitHub pull request using the gh CLI. Requires GITHUB_TOKEN environment variable."
    )
    require_approval = False
    parameters = [
        ToolParameter(
            name="action",
            type="string",
            description="Action: 'create', 'view', 'update_body'.",
            required=True,
            enum=["create", "view", "update_body"],
        ),
        ToolParameter(
            name="repo",
            type="string",
            description="Repository in owner/repo format (used with action=create).",
            required=False,
            default="",
        ),
        ToolParameter(
            name="pr_url",
            type="string",
            description="Full URL or number of the PR (required for action=view or action=update_body).",
            required=False,
            default="",
        ),
        ToolParameter(
            name="title",
            type="string",
            description="PR title (required for action=create).",
            required=False,
            default="",
        ),
        ToolParameter(
            name="body",
            type="string",
            description="PR description body (used for action=create or action=update_body).",
            required=False,
            default="",
        ),
        ToolParameter(
            name="base",
            type="string",
            description="Base branch for the PR (used with action=create). Defaults to 'main'.",
            required=False,
            default="main",
        ),
        ToolParameter(
            name="head",
            type="string",
            description="Head branch for the PR (used with action=create). Defaults to current branch.",
            required=False,
            default="",
        ),
    ]

    async def execute(
        self,
        *,
        action: str,
        repo: str = "",
        pr_url: str = "",
        title: str = "",
        body: str = "",
        base: str = "main",
        head: str = "",
    ) -> ToolResult:
        """Execute a GitHub PR action via gh CLI."""
        token = os.getenv("GITHUB_TOKEN", "")
        if not token:
            return ToolResult.from_error("GITHUB_TOKEN environment variable is not set")

        env = {**os.environ, "GITHUB_TOKEN": token}

        if action == "create":
            if not title:
                return ToolResult.from_error("title is required for action=create")
            cmd = ["gh", "pr", "create", "--title", title, "--base", base]
            if body:
                cmd += ["--body", body]
            if head:
                cmd += ["--head", head]
            if repo:
                cmd += ["--repo", repo]
        elif action == "view":
            if not pr_url:
                return ToolResult.from_error("pr_url is required for action=view")
            cmd = ["gh", "pr", "view", pr_url]
        elif action == "update_body":
            if not pr_url:
                return ToolResult.from_error("pr_url is required for action=update_body")
            cmd = ["gh", "pr", "edit", pr_url, "--body", body]
        else:
            return ToolResult.from_error(f"Unknown action: {action}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            out = stdout.decode(errors="replace")
            err = stderr.decode(errors="replace")
            if proc.returncode == 0:
                return ToolResult.success(
                    output=out or f"Action '{action}' completed successfully.",
                    data={"action": action, "return_code": proc.returncode},
                )
            return ToolResult.from_error(
                f"gh exited with code {proc.returncode}: {err}",
                output=out,
            )
        except FileNotFoundError:
            return ToolResult.from_error("'gh' CLI not found. Install the GitHub CLI to use this tool.")
        except TimeoutError:
            return ToolResult.from_error("gh command timed out")
        except OSError as exc:
            return ToolResult.from_error(f"Error running gh: {exc}")


_ASKCC_DEFAULT_TIMEOUT = 10800  # 3 hours


class AskccRunTool(BaseTool):
    """Run askcc CLI actions on GitHub issue URLs."""

    name = "askcc_run"
    description = (
        "Invoke askcc CLI actions (prepare, develop, review, diagnose, explore, plan) "
        "on a GitHub issue URL. Requires the askcc CLI to be installed."
    )
    require_approval = True
    parameters = [
        ToolParameter(
            name="action",
            type="string",
            description="Action to run: 'prepare', 'develop', 'review', 'diagnose', 'explore', 'plan'.",
            required=True,
            enum=["prepare", "develop", "review", "diagnose", "explore", "plan"],
        ),
        ToolParameter(
            name="issue_url",
            type="string",
            description="Full URL of the GitHub issue (e.g. https://github.com/owner/repo/issues/1).",
            required=True,
        ),
        ToolParameter(
            name="config",
            type="object",
            description="Optional configuration. Supports 'timeout' (seconds, default 10800).",
            required=False,
            default=None,
        ),
    ]

    async def execute(
        self,
        action: str,
        issue_url: str,
        config: dict | None = None,
    ) -> ToolResult:
        """Run askcc {action} {issue_url} via subprocess."""
        timeout = _ASKCC_DEFAULT_TIMEOUT
        if config and "timeout" in config:
            timeout = int(config["timeout"])

        cmd = ["askcc", action, issue_url]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except TimeoutError:
                proc.terminate()
                await proc.wait()
                return ToolResult.from_error(f"askcc {action} timed out after {timeout}s")
            out = stdout.decode(errors="replace")
            err = stderr.decode(errors="replace")
            if proc.returncode == 0:
                return ToolResult.success(
                    output=out or f"askcc {action} completed successfully.",
                    data={"action": action, "issue_url": issue_url, "return_code": proc.returncode},
                )
            return ToolResult.from_error(
                f"askcc exited with code {proc.returncode}: {err}",
                output=out,
            )
        except FileNotFoundError:
            return ToolResult.from_error("'askcc' CLI not found. Install the askcc CLI to use this tool.")
        except OSError as exc:
            return ToolResult.from_error(f"Error running askcc: {exc}")


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
        BrowserAutomationTool(),
        GitHubIssueTool(),
        GitHubPRTool(),
        AskccRunTool(),
    ]
    for tool in tools:
        try:
            registry.register(tool)
        except ValueError:
            logger.debug("Tool %s already registered", tool.name)
