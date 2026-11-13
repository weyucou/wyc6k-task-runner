# Copyright (c) 2026 MNKT
import datetime as dt
import logging
import re
from typing import Any
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from pydantic import BaseModel

from marvin.functions import get_s3_client
from marvin.settings import MEMORY_RECENT_ENTRIES

logger = logging.getLogger(__name__)

_MEMORY_FILE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})(_WEEKLY_SUMMARY)?\.md$")


class MemoryEntry(BaseModel):
    date: dt.date
    filename: str
    content: str
    is_weekly_summary: bool = False


class CustomerContextBundle(BaseModel):
    customer_id: str
    claude_md: str
    sops: dict[str, str]
    project_goals: str
    memory_index: str
    daily_memories: list[MemoryEntry]


class ContextBundleService:
    def _read_object(self, s3: Any, bucket: str, key: str, default: str = "") -> str:
        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            return body.decode("utf-8")
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return default
            raise

    def pull(self, s3_prefix: str) -> CustomerContextBundle:
        """Read the context files from S3 and return a CustomerContextBundle.

        Reads CLAUDE.md, the SOPs, the project README.md and MEMORY.md, plus the newest
        MEMORY_RECENT_ENTRIES memory files (newest first), so the reads stay bounded as memory grows.

        s3_prefix must point to the project directory, e.g.:
        s3://bucket/{customer_prefix}/projects/{repo_name}/
        """
        parsed = urlparse(s3_prefix)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/").rstrip("/")

        # Derive customer root: split on '/projects/' to find the customer prefix.
        # Structure: {customer_prefix}/projects/{repo_name}
        parts = prefix.split("/projects/", 1)
        customer_prefix = parts[0]
        customer_id = customer_prefix.split("/")[-1]

        s3 = get_s3_client()
        paginator = s3.get_paginator("list_objects_v2")

        claude_md = self._read_object(s3, bucket, f"{customer_prefix}/CLAUDE.md")

        sops: dict[str, str] = {}
        sops_prefix = f"{customer_prefix}/sops/"
        for page in paginator.paginate(Bucket=bucket, Prefix=sops_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = key[len(sops_prefix) :]
                if filename:
                    sops[filename] = self._read_object(s3, bucket, key)

        project_goals = self._read_object(s3, bucket, f"{prefix}/README.md")
        memory_index = self._read_object(s3, bucket, f"{prefix}/MEMORY.md")

        memory_files: list[tuple[dt.date, str, str, bool]] = []  # (date, filename, key, is_weekly_summary)
        memory_prefix = f"{prefix}/memory/"
        for page in paginator.paginate(Bucket=bucket, Prefix=memory_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = key.split("/")[-1]
                match = _MEMORY_FILE_PATTERN.search(filename)
                if match:
                    entry_date = dt.date.fromisoformat(match.group(1))
                    memory_files.append((entry_date, filename, key, bool(match.group(2))))
        memory_files.sort(reverse=True)  # newest date first
        daily_memories = [
            MemoryEntry(
                date=entry_date,
                filename=filename,
                content=self._read_object(s3, bucket, key),
                is_weekly_summary=is_weekly_summary,
            )
            for entry_date, filename, key, is_weekly_summary in memory_files[:MEMORY_RECENT_ENTRIES]
        ]

        return CustomerContextBundle(
            customer_id=customer_id,
            claude_md=claude_md,
            sops=sops,
            project_goals=project_goals,
            memory_index=memory_index,
            daily_memories=daily_memories,
        )

    def push_memory(self, s3_prefix: str, entry: MemoryEntry) -> None:
        """Append entry content to the daily memory file on S3, creating if missing."""
        parsed = urlparse(s3_prefix)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/").rstrip("/")

        memory_key = f"{prefix}/memory/{entry.date.year}/{entry.filename}"

        s3 = get_s3_client()
        existing = self._read_object(s3, bucket, memory_key, default="")
        if existing and not existing.endswith("\n"):
            existing += "\n"
        new_content = f"{existing}{entry.content}" if existing else entry.content

        s3.put_object(
            Bucket=bucket,
            Key=memory_key,
            Body=new_content.encode("utf-8"),
            ContentType="text/markdown",
        )
        logger.info("Wrote memory entry to s3://%s/%s", bucket, memory_key)

