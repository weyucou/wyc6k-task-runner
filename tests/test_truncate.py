# Copyright (c) 2026 MNKT
"""Tests for the tool output cap (`marvin/tools/truncate.py`)."""

import re
import runpy
import stat
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from marvin import settings
from marvin.tools import truncate
from marvin.tools.truncate import (
    OutputCapture,
    SpillDir,
    cap_head,
    cleanup_spill_dir,
    measure,
    truncate_head,
    truncate_tail,
)
from tests.notices import HEAD_NOTICE, TAIL_NOTICE

NUMBERS = "".join(f"{n}\n" for n in range(1, 101))  # "1\n2\n...100\n"


def numbered(count: int) -> str:
    """One line per number, from 1 to `count`."""
    return "".join(f"{n}\n" for n in range(1, count + 1))


def captured(data: bytes, *, chunk: int = 4096, max_lines: int = 100, max_bytes: int = 2000) -> OutputCapture:
    """An OutputCapture fed `data` in `chunk`-byte writes."""
    capture = OutputCapture("exec", max_lines=max_lines, max_bytes=max_bytes)
    for start in range(0, len(data), chunk):
        capture.write(data[start : start + chunk])
    return capture


@pytest.fixture
def spill_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create spill directories under tmp_path."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


class TestSettings:
    @pytest.fixture(autouse=True)
    def _no_cap_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MARVIN_TOOL_OUTPUT_MAX_LINES", raising=False)
        monkeypatch.delenv("MARVIN_TOOL_OUTPUT_MAX_BYTES", raising=False)

    def test_defaults_are_2000_lines_and_50_kib(self) -> None:
        module = runpy.run_path(settings.__file__)
        assert (module["TOOL_OUTPUT_MAX_LINES"], module["TOOL_OUTPUT_MAX_BYTES"]) == (2000, 51200)

    def test_read_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MARVIN_TOOL_OUTPUT_MAX_LINES", "300")
        monkeypatch.setenv("MARVIN_TOOL_OUTPUT_MAX_BYTES", "4096")
        module = runpy.run_path(settings.__file__)
        assert (module["TOOL_OUTPUT_MAX_LINES"], module["TOOL_OUTPUT_MAX_BYTES"]) == (300, 4096)

    @pytest.mark.parametrize(
        ("name", "value", "error"),
        [
            ("MARVIN_TOOL_OUTPUT_MAX_LINES", "9", "MARVIN_TOOL_OUTPUT_MAX_LINES must be at least 10, got 9"),
            ("MARVIN_TOOL_OUTPUT_MAX_BYTES", "1023", "MARVIN_TOOL_OUTPUT_MAX_BYTES must be at least 1024, got 1023"),
            ("MARVIN_TOOL_OUTPUT_MAX_BYTES", "50KB", "invalid literal for int"),
        ],
        ids=["lines-below-floor", "bytes-below-floor", "not-an-integer"],
    )
    def test_a_bad_value_fails_at_import(
        self, monkeypatch: pytest.MonkeyPatch, name: str, value: str, error: str
    ) -> None:
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match=error):
            runpy.run_path(settings.__file__)


class TestMeasure:
    @pytest.mark.parametrize(
        ("text", "lines", "size"),
        [
            ("", 0, 0),
            ("a", 1, 1),
            ("a\n", 1, 2),
            ("a\nb", 2, 3),
            ("\n\n", 2, 2),
            ("あ", 1, 3),
        ],
        ids=["empty", "no-newline", "final-newline", "two-lines", "blank-lines", "multibyte"],
    )
    def test_counts_lines_and_utf8_bytes(self, text: str, lines: int, size: int) -> None:
        assert measure(text) == (lines, size)


class TestTruncateHead:
    def test_text_within_the_limits_is_unchanged(self) -> None:
        assert truncate_head("a\nb\n", max_lines=2, max_bytes=4) == "a\nb\n"

    def test_keeps_the_first_lines(self) -> None:
        assert truncate_head(NUMBERS, max_lines=3, max_bytes=1000) == "1\n2\n3\n"

    def test_cuts_after_the_last_whole_line_within_the_bytes(self) -> None:
        assert truncate_head("aaaa\nbbbb\ncccc\n", max_lines=10, max_bytes=12) == "aaaa\nbbbb\n"

    def test_a_first_line_over_the_bytes_is_cut_inside(self) -> None:
        assert truncate_head("x" * 100 + "\nshort\n", max_lines=10, max_bytes=10) == "x" * 10

    def test_never_splits_a_multibyte_character(self) -> None:
        # "あ" is 3 bytes in UTF-8: 10 bytes hold 3 whole characters.
        assert truncate_head("あ" * 10, max_lines=10, max_bytes=10) == "あ" * 3


class TestTruncateTail:
    def test_text_within_the_limits_is_unchanged(self) -> None:
        assert truncate_tail("a\nb\n", max_lines=2, max_bytes=4) == "a\nb\n"

    def test_keeps_the_last_lines(self) -> None:
        assert truncate_tail(NUMBERS, max_lines=3, max_bytes=1000) == "98\n99\n100\n"

    def test_cuts_before_the_first_whole_line_within_the_bytes(self) -> None:
        assert truncate_tail("aaaa\nbbbb\ncccc\n", max_lines=10, max_bytes=12) == "bbbb\ncccc\n"

    def test_a_last_line_over_the_bytes_keeps_its_end(self) -> None:
        assert truncate_tail("short\n" + "x" * 90 + "y" * 10, max_lines=10, max_bytes=10) == "y" * 10

    def test_never_splits_a_multibyte_character(self) -> None:
        assert truncate_tail("あ" * 10, max_lines=10, max_bytes=10) == "あ" * 3


class TestSpillDir:
    def test_created_on_first_use_and_owner_only(self, spill_root: Path) -> None:
        spill = SpillDir()
        assert list(spill_root.iterdir()) == []
        path, file = spill.open("exec")
        with file:
            file.write(b"x")
        assert path.parent.parent == spill_root
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_each_run_gets_its_own_directory(self, spill_root: Path) -> None:
        spill = SpillDir()
        first, file = spill.open("exec")
        file.close()
        spill.cleanup()
        second, file = spill.open("exec")
        file.close()
        assert not first.parent.exists()
        assert second.parent != first.parent
        assert second.parent.parent == spill_root

    def test_names_are_sanitised(self, spill_root: Path) -> None:
        path, file = SpillDir().open("../../etc/pass wd")
        file.close()
        assert path.parent.parent == spill_root
        assert re.fullmatch(r"[A-Za-z0-9_-]+-\d+\.log", path.name)

    def test_refuses_a_planted_symlink(self, spill_root: Path, tmp_path: Path) -> None:
        spill = SpillDir()
        first, file = spill.open("exec")
        file.close()
        target = tmp_path / "target"
        target.write_text("keep", encoding="utf-8")
        (first.parent / "exec-2.log").symlink_to(target)  # the next name `open` picks
        with pytest.raises(FileExistsError):
            spill.open("exec")
        assert target.read_text(encoding="utf-8") == "keep"

    def test_cleanup_spill_dir_removes_the_run_directory(self, spill_root: Path) -> None:
        path = Path(HEAD_NOTICE.search(cap_head(numbered(100), max_lines=10, max_bytes=2000))[5])
        assert path.exists()
        cleanup_spill_dir()
        assert not path.parent.exists()
        cleanup_spill_dir()  # a second call is a no-op


class TestCapHead:
    def test_small_text_is_unchanged_and_creates_no_spill_dir(self, spill_root: Path) -> None:
        assert cap_head("hello\n") == "hello\n"
        assert list(spill_root.iterdir()) == []

    @pytest.mark.parametrize(
        ("max_lines", "max_bytes"),
        [(100, 100_000), (100_000, 2000)],
        ids=["over-the-lines", "over-the-bytes"],
    )
    def test_keeps_the_head_and_spills_the_full_text(self, spill_root: Path, max_lines: int, max_bytes: int) -> None:
        text = numbered(10_000)
        capped = cap_head(text, name="web_fetch", max_lines=max_lines, max_bytes=max_bytes)
        lines, size = measure(capped)
        assert lines <= max_lines
        assert size <= max_bytes
        *head, blank, notice = capped.split("\n")
        match = HEAD_NOTICE.fullmatch(notice)
        assert match is not None
        shown, total, shown_bytes, total_bytes, path = match.groups()
        assert blank == ""
        assert head == [str(n) for n in range(1, int(shown) + 1)]
        assert (int(total), int(total_bytes)) == measure(text)
        assert int(shown_bytes) == len("\n".join(head)) + 1
        assert Path(path).read_text(encoding="utf-8") == text
        assert Path(path).name.startswith("web_fetch-")

    def test_never_splits_a_multibyte_character(self, spill_root: Path) -> None:
        capped = cap_head("あ" * 100_000, max_lines=100, max_bytes=2000)
        head, notice = capped.split("\n\n")
        assert set(head) == {"あ"}
        assert measure(capped)[1] <= 2000
        assert HEAD_NOTICE.fullmatch(notice)

    def test_leaves_room_for_the_reserve(self, spill_root: Path) -> None:
        reserve = "\n\nError: boom"
        capped = cap_head(numbered(10_000), reserve=reserve, max_lines=100, max_bytes=2000)
        lines, size = measure(capped + reserve)
        assert lines <= 100
        assert size <= 2000

    def test_text_that_fits_only_without_the_reserve_is_cut(self, spill_root: Path) -> None:
        text = numbered(95)
        assert cap_head(text, max_lines=100, max_bytes=2000) == text
        assert HEAD_NOTICE.search(cap_head(text, reserve="\n" * 10, max_lines=100, max_bytes=2000))

    def test_is_idempotent(self, spill_root: Path) -> None:
        once = cap_head(numbered(10_000), max_lines=100, max_bytes=2000)
        assert cap_head(once, max_lines=100, max_bytes=2000) == once

    def test_a_failed_spill_still_caps(self, spill_root: Path) -> None:
        with patch.object(truncate._SPILL_DIR, "open", side_effect=OSError("disk full")):
            capped = cap_head(numbered(10_000), max_lines=100, max_bytes=2000)
        assert measure(capped)[0] <= 100
        assert capped.endswith("bytes). Full output could not be saved]")


class TestOutputCapture:
    def test_small_output_is_whole_and_creates_no_spill_dir(self, spill_root: Path) -> None:
        with OutputCapture("exec") as capture:
            capture.write(b"hello\n")
            assert capture.render() == "hello\n"
        assert (capture.truncated, capture.path) == (False, None)
        assert list(spill_root.iterdir()) == []

    def test_keeps_the_tail_and_spills_everything(self, spill_root: Path) -> None:
        data = numbered(100_000).encode()
        with captured(data) as capture:
            rendered = capture.render()
        lines, size = measure(rendered)
        assert lines <= 100
        assert size <= 2000
        *tail, blank, notice = rendered.split("\n")
        match = TAIL_NOTICE.fullmatch(notice)
        assert match is not None
        shown, total, shown_bytes, total_bytes, path = match.groups()
        assert blank == ""
        assert tail == [str(n) for n in range(100_001 - int(shown), 100_001)]
        assert (int(total), int(total_bytes)) == (100_000, len(data))
        assert int(shown_bytes) == len("\n".join(tail)) + 1
        assert Path(path).read_bytes() == data
        assert (capture.truncated, capture.path) == (True, Path(path))

    def test_memory_stays_bounded(self, spill_root: Path) -> None:
        line = b"x" * 1023 + b"\n"
        with OutputCapture("exec", max_bytes=4096) as capture:
            for _ in range(10 * 1024):  # 10 MiB
                capture.write(line)
                assert len(capture._window) <= 4096
        assert capture.path is not None
        assert capture.path.stat().st_size == 10 * 1024 * 1024

    def test_leaves_room_for_the_reserve(self, spill_root: Path) -> None:
        reserve = "\n\nError: Command exited with code 1."
        with captured(numbered(10_000).encode()) as capture:
            rendered = capture.render(reserve=reserve)
        lines, size = measure(rendered + reserve)
        assert lines <= 100
        assert size <= 2000
        assert TAIL_NOTICE.search(rendered)

    def test_output_over_the_cap_only_with_the_reserve_is_cut_and_spilled(self, spill_root: Path) -> None:
        data = numbered(95).encode()
        with captured(data) as capture:
            rendered = capture.render(reserve="\n" * 10)
        match = TAIL_NOTICE.search(rendered)
        assert match is not None
        assert Path(match[5]).read_bytes() == data

    def test_a_character_split_across_writes_is_kept(self) -> None:
        with OutputCapture("exec") as capture:
            capture.write("あ".encode()[:2])
            capture.write("あ\n".encode()[2:])
            assert capture.render() == "あ\n"

    def test_invalid_utf8_is_replaced(self) -> None:
        with OutputCapture("exec") as capture:
            capture.write(b"caf\xe9\n")
            assert capture.render() == "caf\ufffd\n"

    def test_writes_after_close_are_dropped(self, spill_root: Path) -> None:
        capture = OutputCapture("process")
        capture.write(b"one\n")
        capture.close()
        capture.write(b"x" * 100_000)
        assert capture.render() == "one\n"
        assert list(spill_root.iterdir()) == []

    def test_a_failed_spill_keeps_the_tail(self, spill_root: Path) -> None:
        with (
            patch.object(truncate._SPILL_DIR, "open", side_effect=OSError("disk full")),
            captured(numbered(10_000).encode()) as capture,
        ):
            rendered = capture.render()
        assert rendered.endswith("bytes). Full output could not be saved]")
        assert measure(rendered)[0] <= 100
        assert capture.path is None
