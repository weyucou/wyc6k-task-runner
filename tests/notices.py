# Copyright (c) 2026 MNKT
"""Patterns for the truncation notices that end a capped tool result (`marvin/tools/truncate.py`)."""

import re

# Groups: shown lines, total lines, shown bytes, total bytes, spill file.
HEAD_NOTICE = re.compile(r"\[Showing lines 1-(\d+) of (\d+) \((\d+) of (\d+) bytes\)\. Full output saved to (.+)\]")
TAIL_NOTICE = re.compile(r"\[Showing last (\d+) of (\d+) lines \((\d+) of (\d+) bytes\)\. Full output saved to (.+)\]")
