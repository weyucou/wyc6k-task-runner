# Copyright (c) 2026 MNKT
"""The tool output cap: how much of a tool result the model sees, and the spill files that keep the rest.

Every result is capped at TOOL_OUTPUT_MAX_LINES lines and TOOL_OUTPUT_MAX_BYTES UTF-8 bytes, notice included
(`marvin/settings.py`). `cap_head` keeps the head of a text. Over the cap, the full text goes to a spill file in a
per-run directory, created on first use and deleted by `cleanup_spill_dir()` when the run ends.
"""

import contextlib
import logging
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import BinaryIO, Self

from marvin.settings import TOOL_OUTPUT_MAX_BYTES, TOOL_OUTPUT_MAX_LINES

logger = logging.getLogger(__name__)

# The most `web_fetch` and `browser_fetch` read of a response body. It sits far above the cap because stripping
# HTML shrinks a page a lot.
FETCH_MAX_BYTES = 5 * 1024 * 1024
_SPILL_NAME_MAX_CHARS = 64


def _encode(text: str) -> bytes:
    return text.encode("utf-8", errors="replace")


def _count_lines(data: bytes) -> int:
    """A final newline ends the last line; it does not start a new one."""
    return data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)


def measure(text: str) -> tuple[int, int]:
    """Lines and UTF-8 bytes in `text`."""
    data = _encode(text)
    return _count_lines(data), len(data)


def fits(text: str, *, max_lines: int = TOOL_OUTPUT_MAX_LINES, max_bytes: int = TOOL_OUTPUT_MAX_BYTES) -> bool:
    """True when `text` is within the cap."""
    lines, size = measure(text)
    return lines <= max_lines and size <= max_bytes


def _head_end(data: bytes, max_lines: int, max_bytes: int) -> int:
    """Where the head of `data` ends: after its last whole line within the limits."""
    if max_lines <= 0 or max_bytes <= 0:
        return 0
    end = 0
    for _ in range(max_lines):
        newline = data.find(b"\n", end, max_bytes)
        if newline == -1:
            if len(data) <= max_bytes:
                end = len(data)  # the rest is a last line without a newline, and it fits
            break
        end = newline + 1
    return end or max_bytes  # a first line over max_bytes is cut inside


def _tail_start(data: bytes, max_lines: int, max_bytes: int) -> int:
    """Where the tail of `data` starts: at its first whole line within the limits."""
    if max_lines <= 0 or max_bytes <= 0:
        return len(data)
    floor = max(len(data) - max_bytes, 0)  # the earliest start the byte limit allows
    start = len(data)
    end = len(data) - 1 if data.endswith(b"\n") else len(data)  # a final newline belongs to the last line
    for _ in range(max_lines):
        newline = data.rfind(b"\n", max(floor - 1, 0), end)
        if newline == -1:
            if floor == 0:
                start = 0  # the rest is the first line, and it fits
            break
        start = newline + 1
        end = newline
    return floor if start == len(data) else start  # a last line over max_bytes is cut inside


def truncate_head(text: str, *, max_lines: int, max_bytes: int) -> str:
    """The longest head of `text` within `max_lines` and `max_bytes`, cut after a line.

    Only a first line longer than `max_bytes` is cut inside, never inside a UTF-8 character.
    """
    data = _encode(text)
    return data[: _head_end(data, max_lines, max_bytes)].decode("utf-8", errors="ignore")


def truncate_tail(text: str, *, max_lines: int, max_bytes: int) -> str:
    """The longest tail of `text` within `max_lines` and `max_bytes`, cut before a line.

    Only a last line longer than `max_bytes` is cut inside, never inside a UTF-8 character.
    """
    data = _encode(text)
    return data[_tail_start(data, max_lines, max_bytes) :].decode("utf-8", errors="ignore")


class SpillDir:
    """A run's spill files, in a directory only its owner can read, created on first use."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._count = 0
        self._lock = threading.Lock()  # `process` reader threads spill too

    def open(self, name: str) -> tuple[Path, BinaryIO]:
        """Create a new spill file for tool `name`. Raises OSError, also when the path already exists."""
        with self._lock:
            if self._path is None:
                self._path = Path(tempfile.mkdtemp(prefix="marvin-spill-"))  # unique per run, mode 0o700
            self._count += 1
            path = self._path / f"{_safe_name(name)}-{self._count}.log"
        # O_EXCL refuses any existing entry, a planted symlink included; O_NOFOLLOW states the intent.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        return path, os.fdopen(fd, "wb")

    def cleanup(self) -> None:
        """Delete the directory and its spill files; the next spill creates a new directory."""
        with self._lock:
            path, self._path = self._path, None
        if path is not None:
            shutil.rmtree(path, ignore_errors=True)


_SPILL_DIR = SpillDir()  # one run at a time per process, like the `process` sessions


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)[:_SPILL_NAME_MAX_CHARS]


def _spill_text(name: str, text: str) -> Path | None:
    """Save `text` to a new spill file; None, logged, when that fails."""
    try:
        path, file = _SPILL_DIR.open(name)
        with file:
            file.write(_encode(text))
    except OSError:
        logger.warning("Could not save the full %s output to a spill file", name, exc_info=True)
        return None
    return path


def _saved(path: Path | None) -> str:
    return f"Full output saved to {path}" if path else "Full output could not be saved"


def _head_notice(shown_lines: int, total_lines: int, shown_bytes: int, total_bytes: int, path: Path | None) -> str:
    return f"[Showing lines 1-{shown_lines} of {total_lines} ({shown_bytes} of {total_bytes} bytes). {_saved(path)}]"


def _tail_notice(shown_lines: int, total_lines: int, shown_bytes: int, total_bytes: int, path: Path | None) -> str:
    return f"[Showing last {shown_lines} of {total_lines} lines ({shown_bytes} of {total_bytes} bytes). {_saved(path)}]"


def _with_notice(content: str, notice: str) -> str:
    """`content`, a blank line, then `notice`."""
    content = content.removesuffix("\n")
    return f"{content}\n\n{notice}" if content else notice


def cap_head(
    text: str,
    *,
    name: str = "output",
    reserve: str = "",
    max_lines: int = TOOL_OUTPUT_MAX_LINES,
    max_bytes: int = TOOL_OUTPUT_MAX_BYTES,
) -> str:
    """`text` whole when it fits the cap; otherwise its head, a blank line and a notice naming the spill file.

    Args:
        text: The tool output.
        name: The tool name, for the spill file name.
        reserve: Text the caller appends to the result; the result leaves room for it.
        max_lines: The line cap.
        max_bytes: The UTF-8 byte cap.
    """
    if fits(text + reserve, max_lines=max_lines, max_bytes=max_bytes):
        return text
    total_lines, total_bytes = measure(text)
    path = _spill_text(name, text)

    def notice(shown_lines: int, shown_bytes: int) -> str:
        return _head_notice(shown_lines, total_lines, shown_bytes, total_bytes, path)

    # Leave room for the reserve and for the longest notice this text can get (everything shown).
    reserve_lines, reserve_bytes = measure(reserve)
    notice_lines, notice_bytes = measure("\n\n" + notice(total_lines, total_bytes))
    head = truncate_head(
        text,
        max_lines=max_lines - reserve_lines - notice_lines,
        max_bytes=max_bytes - reserve_bytes - notice_bytes,
    )
    return _with_notice(head, notice(*measure(head)))


def read_body(response: BinaryIO) -> tuple[bytes, str]:
    """At most FETCH_MAX_BYTES of a response body, and a note to append to the result when the body is longer."""
    body = response.read(FETCH_MAX_BYTES + 1)
    if len(body) <= FETCH_MAX_BYTES:
        return body, ""
    return body[:FETCH_MAX_BYTES], f"\n\n[Only the first {FETCH_MAX_BYTES} bytes of the response body were read.]"


def cleanup_spill_dir() -> None:
    """Delete the run's spill directory; call when an agent run ends, after `stop_background_processes()`."""
    _SPILL_DIR.cleanup()


class OutputCapture:
    """A command's output in bounded memory: its tail in memory and, once over the cap, all of it in a spill file.

    `write` takes the output as it arrives; `render` gives the text the model sees. Writes after `close` are dropped,
    so a closed capture never creates a spill file.
    """

    def __init__(
        self, name: str, *, max_lines: int = TOOL_OUTPUT_MAX_LINES, max_bytes: int = TOOL_OUTPUT_MAX_BYTES
    ) -> None:
        self.total_bytes = 0
        self.truncated = False  # over the cap: `render` shows the tail
        self.path: Path | None = None  # the spill file with the full output
        self._name = name
        self._max_lines = max_lines
        self._max_bytes = max_bytes
        self._newlines = 0
        self._window = bytearray()  # the whole output until it is over the cap, then its last max_bytes
        self._file: BinaryIO | None = None
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def total_lines(self) -> int:
        return self._newlines + (1 if self._window and not self._window.endswith(b"\n") else 0)

    def write(self, chunk: bytes) -> None:
        """Add `chunk` to the output."""
        if self._closed:
            return
        self.total_bytes += len(chunk)
        self._newlines += chunk.count(b"\n")
        self._window += chunk
        if self._file is not None:
            self._spill(chunk)
        elif not self.truncated and (self.total_bytes > self._max_bytes or self.total_lines > self._max_lines):
            self._start_spill()
        if self.truncated and len(self._window) > self._max_bytes:
            del self._window[: len(self._window) - self._max_bytes]

    def render(self, reserve: str = "") -> str:
        """The text the model sees, leaving room for `reserve`: the output whole if it fits, else its tail and a notice.

        Args:
            reserve: Text the caller appends to the result.
        """
        text = self._window.decode("utf-8", errors="replace")
        if not self.truncated:
            if fits(text + reserve, max_lines=self._max_lines, max_bytes=self._max_bytes):
                return text
            self._start_spill()  # only the reserve takes it over the cap; the window still holds all of it
        reserve_lines, reserve_bytes = measure(reserve)
        notice_lines, notice_bytes = measure("\n\n" + self._notice(self.total_lines, self.total_bytes))
        tail = truncate_tail(
            text,
            max_lines=self._max_lines - reserve_lines - notice_lines,
            max_bytes=self._max_bytes - reserve_bytes - notice_bytes,
        )
        return _with_notice(tail, self._notice(*measure(tail)))

    def close(self) -> None:
        """Stop capturing and close the spill file."""
        self._closed = True
        if self._file is not None:
            self._file.close()
            self._file = None

    def _notice(self, shown_lines: int, shown_bytes: int) -> str:
        return _tail_notice(shown_lines, self.total_lines, shown_bytes, self.total_bytes, self.path)

    def _start_spill(self) -> None:
        """Mark the output as over the cap and save the window, which holds all of it so far, to a spill file."""
        self.truncated = True
        if self._closed:
            return
        try:
            self.path, self._file = _SPILL_DIR.open(self._name)
        except OSError:
            logger.warning("Could not save the full %s output to a spill file", self._name, exc_info=True)
            return
        self._spill(bytes(self._window))

    def _spill(self, data: bytes) -> None:
        """Append `data` to the spill file; on an OS error, give up on the file, since it would be incomplete."""
        if self._file is None:
            return
        try:
            self._file.write(data)
            self._file.flush()  # the model may read the file while the command still runs
        except OSError:
            logger.warning("Could not save the full %s output to a spill file", self._name, exc_info=True)
            with contextlib.suppress(OSError):  # closing flushes what the failed write left buffered
                self._file.close()
            self._file = None
            self.path = None
