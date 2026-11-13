# Copyright (c) 2026 MNKT
"""Coding tools for file operations, shell execution, web fetching, and GitHub."""

import asyncio
import logging
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any
from urllib.request import Request, urlopen

from marvin.tools.base import BaseTool, ToolParameter, ToolResult, ToolStatus

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


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
        "Use this to call any installed CLI — gh, git, or any other tool on PATH. "
        "Prefer this over specialised wrappers when a CLI already exists for the task. "
        "Commands run with stdin closed and are terminated on timeout; "
        "start servers and other background jobs with the `process` tool. "
        "Increase timeout for long-running tools (e.g. a full test suite or a large build)."
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
        GitHubIssueTool(),
        GitHubPRTool(),
    ]
    for tool in tools:
        try:
            registry.register(tool)
        except ValueError:
            logger.debug("Tool %s already registered", tool.name)
