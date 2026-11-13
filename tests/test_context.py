# Copyright (c) 2026 MNKT
"""Tests for marvin.context — ContextBundleService against moto-backed S3."""

from typing import TYPE_CHECKING, Any

import boto3
import pytest
from moto import mock_aws

from marvin import context
from marvin.context import ContextBundleService

if TYPE_CHECKING:
    from collections.abc import Iterator

BUCKET = "agent-context"
PROJECT_KEY = "customers/acme/projects/repo"
S3_CONTEXT_PREFIX = f"s3://{BUCKET}/{PROJECT_KEY}"


class S3Bucket:
    """A moto bucket that `marvin.context` reads, plus the key of every GetObject call made on it."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.read_keys: list[str] = []
        client.meta.events.register("before-parameter-build.s3.GetObject", self._record_read)

    def _record_read(self, params: dict[str, Any], **_: Any) -> None:
        self.read_keys.append(params["Key"])

    def put(self, objects: dict[str, str]) -> None:
        for key, body in objects.items():
            self.client.put_object(Bucket=BUCKET, Key=key, Body=body.encode("utf-8"))


@pytest.fixture
def bucket(monkeypatch) -> Iterator[S3Bucket]:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        monkeypatch.setattr(context, "get_s3_client", lambda: client)
        yield S3Bucket(client)


class TestPullMemory:
    def test_reads_only_the_newest_memory_files(self, bucket, monkeypatch) -> None:
        monkeypatch.setattr(context, "MEMORY_RECENT_ENTRIES", 3)
        memory = f"{PROJECT_KEY}/memory"
        bucket.put(
            {
                f"{PROJECT_KEY}/MEMORY.md": "index",
                f"{memory}/2025/2025-12-31.md": "last year",
                f"{memory}/2026/2026-01-01.md": "new year",
                f"{memory}/2026/2026-01-02.md": "second",
                f"{memory}/2026/2026-01-02_WEEKLY_SUMMARY.md": "week",
                f"{memory}/2026/2026-01-03.md": "third",
                f"{memory}/2026/notes.txt": "not a memory file",
            }
        )

        bundle = ContextBundleService().pull(S3_CONTEXT_PREFIX)

        assert bundle.memory_index == "index"
        assert [(entry.filename, entry.content, entry.is_weekly_summary) for entry in bundle.daily_memories] == [
            ("2026-01-03.md", "third", False),
            ("2026-01-02_WEEKLY_SUMMARY.md", "week", True),
            ("2026-01-02.md", "second", False),
        ]
        memory_reads = sorted(key for key in bucket.read_keys if key.startswith(f"{memory}/"))
        assert memory_reads == [
            f"{memory}/2026/2026-01-02.md",
            f"{memory}/2026/2026-01-02_WEEKLY_SUMMARY.md",
            f"{memory}/2026/2026-01-03.md",
        ]
