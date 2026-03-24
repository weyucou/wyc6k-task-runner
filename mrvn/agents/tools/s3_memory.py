"""S3-backed memory write tool for agent session notes."""

import datetime
import logging
from typing import Any

from agents.context import ContextBundleService, MemoryEntry
from agents.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class S3MemoryWriteTool(BaseTool):
    """Write agent session notes to the daily memory file on S3."""

    name = "s3_memory_write"
    description = (
        "Append a markdown block to the daily memory file on S3. "
        "Use this to persist what you did, decisions made, and issues updated during this session."
    )
    parameters = [
        ToolParameter(
            name="section_header",
            type="string",
            description="Markdown section heading for this entry, e.g. '## 2026-03-18 14:30:00 Actions'.",
            required=True,
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Markdown block to append under the section header.",
            required=True,
        ),
    ]

    def __init__(self, s3_prefix: str = "", config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.s3_prefix = s3_prefix

    async def execute(self, section_header: str, content: str) -> ToolResult:
        if not self.s3_prefix:
            return ToolResult.from_error("s3_prefix is not configured; cannot write memory.")

        today = datetime.datetime.now(tz=datetime.UTC).date()
        filename = f"{today.isoformat()}.md"
        entry_content = f"{section_header}\n{content}\n"

        entry = MemoryEntry(
            date=today,
            filename=filename,
            content=entry_content,
        )

        try:
            ContextBundleService().push_memory(self.s3_prefix, entry)
        except Exception as exc:
            logger.exception("Failed to write memory entry to S3")
            return ToolResult.from_error(f"Failed to write memory: {exc}")

        key_hint = f"memory/{today.year}/{filename}"
        return ToolResult.success(
            output=f"Memory entry written to {key_hint}.",
            data={"s3_prefix": self.s3_prefix, "filename": filename, "date": today.isoformat()},
        )
