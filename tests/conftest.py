# Copyright (c) 2026 MNKT
"""Fixtures for every test."""

from typing import TYPE_CHECKING

import pytest

from marvin.tools.truncate import cleanup_spill_dir

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _remove_spill_dir() -> Iterator[None]:
    """Delete the spill directory a test's capped tool output created, so the next test starts without one."""
    yield
    cleanup_spill_dir()
