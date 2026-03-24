"""Tests for CustomerContextBundle and ContextBundleService — uses LocalStack S3."""

import datetime
import os
import unittest
from typing import Any, ClassVar

from botocore.exceptions import ClientError

from agents.context import ContextBundleService, CustomerContextBundle, MemoryEntry
from commons.functions import get_s3_client

BUCKET = "wyc6k-agent-contexts-test"
CUSTOMER_PREFIX = "customers/cust-001"
REPO_NAME = "my-repo"
PROJECT_PREFIX = f"s3://{BUCKET}/{CUSTOMER_PREFIX}/projects/{REPO_NAME}"

_AWS_ENV = {
    "AWS_ACCESS_KEY_ID": "test",
    "AWS_SECRET_ACCESS_KEY": "test",
    "AWS_DEFAULT_REGION": "us-east-1",
    "AWS_ENDPOINT_URL": "http://localhost:4566",
}


def _clear_bucket(s3: Any, bucket: str) -> None:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket=bucket, Key=obj["Key"])


class BaseLocalStackTest(unittest.TestCase):
    s3: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.update(_AWS_ENV)
        cls.s3 = get_s3_client()
        cls.s3.create_bucket(Bucket=BUCKET)

    @classmethod
    def tearDownClass(cls) -> None:
        _clear_bucket(cls.s3, BUCKET)
        cls.s3.delete_bucket(Bucket=BUCKET)

    def setUp(self) -> None:
        _clear_bucket(self.s3, BUCKET)


class CustomerContextBundleModelTest(unittest.TestCase):
    def test_memory_entry_defaults(self) -> None:
        entry = MemoryEntry(
            date=datetime.date(2026, 3, 22),
            filename="2026-03-22.md",
            content="content",
        )
        self.assertFalse(entry.is_weekly_summary)

    def test_bundle_fields(self) -> None:
        bundle = CustomerContextBundle(
            customer_id="cust-001",
            claude_md="# Instructions",
            sops={"overview.md": "Safety first"},
            project_goals="# Goals",
            memory_index="# Index",
            daily_memories=[],
        )
        self.assertEqual(bundle.customer_id, "cust-001")
        self.assertEqual(bundle.sops["overview.md"], "Safety first")


class ContextBundleServicePullTest(BaseLocalStackTest):
    def _put(self, key: str, content: str) -> None:
        self.s3.put_object(Bucket=BUCKET, Key=key, Body=content.encode("utf-8"))

    def test_pull_reads_claude_md(self) -> None:
        self._put(f"{CUSTOMER_PREFIX}/CLAUDE.md", "# Claude instructions")
        bundle = ContextBundleService().pull(PROJECT_PREFIX)
        self.assertEqual(bundle.claude_md, "# Claude instructions")

    def test_pull_reads_sops(self) -> None:
        self._put(f"{CUSTOMER_PREFIX}/sops/overview.md", "Safety first")
        self._put(f"{CUSTOMER_PREFIX}/sops/coding.md", "Write tests")
        bundle = ContextBundleService().pull(PROJECT_PREFIX)
        self.assertEqual(bundle.sops["overview.md"], "Safety first")
        self.assertEqual(bundle.sops["coding.md"], "Write tests")

    def test_pull_reads_project_files(self) -> None:
        self._put(f"{CUSTOMER_PREFIX}/projects/{REPO_NAME}/README.md", "Project goals")
        self._put(f"{CUSTOMER_PREFIX}/projects/{REPO_NAME}/MEMORY.md", "# Memory index")
        bundle = ContextBundleService().pull(PROJECT_PREFIX)
        self.assertEqual(bundle.project_goals, "Project goals")
        self.assertEqual(bundle.memory_index, "# Memory index")

    def test_pull_reads_daily_memory_files(self) -> None:
        self._put(f"{CUSTOMER_PREFIX}/projects/{REPO_NAME}/memory/2026/2026-03-22.md", "Daily log")
        bundle = ContextBundleService().pull(PROJECT_PREFIX)
        self.assertEqual(len(bundle.daily_memories), 1)
        entry = bundle.daily_memories[0]
        self.assertEqual(entry.date, datetime.date(2026, 3, 22))
        self.assertEqual(entry.content, "Daily log")
        self.assertFalse(entry.is_weekly_summary)

    def test_pull_reads_weekly_summary_files(self) -> None:
        self._put(
            f"{CUSTOMER_PREFIX}/projects/{REPO_NAME}/memory/2026/2026-03-15_WEEKLY_SUMMARY.md",
            "Weekly summary",
        )
        bundle = ContextBundleService().pull(PROJECT_PREFIX)
        self.assertEqual(len(bundle.daily_memories), 1)
        entry = bundle.daily_memories[0]
        self.assertEqual(entry.date, datetime.date(2026, 3, 15))
        self.assertEqual(entry.content, "Weekly summary")
        self.assertTrue(entry.is_weekly_summary)

    def test_pull_extracts_customer_id(self) -> None:
        bundle = ContextBundleService().pull(PROJECT_PREFIX)
        self.assertEqual(bundle.customer_id, "cust-001")

    def test_pull_missing_files_return_empty_defaults(self) -> None:
        bundle = ContextBundleService().pull(PROJECT_PREFIX)
        self.assertEqual(bundle.claude_md, "")
        self.assertEqual(bundle.sops, {})
        self.assertEqual(bundle.project_goals, "")
        self.assertEqual(bundle.memory_index, "")
        self.assertEqual(bundle.daily_memories, [])

    def test_pull_multiple_memory_files(self) -> None:
        self._put(f"{CUSTOMER_PREFIX}/projects/{REPO_NAME}/memory/2026/2026-03-20.md", "Day 20")
        self._put(f"{CUSTOMER_PREFIX}/projects/{REPO_NAME}/memory/2026/2026-03-21.md", "Day 21")
        self._put(
            f"{CUSTOMER_PREFIX}/projects/{REPO_NAME}/memory/2026/2026-03-15_WEEKLY_SUMMARY.md",
            "Weekly",
        )
        bundle = ContextBundleService().pull(PROJECT_PREFIX)
        self.assertEqual(len(bundle.daily_memories), 3)


class ContextBundleServicePushTest(BaseLocalStackTest):
    def _get(self, key: str) -> str:
        return self.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode("utf-8")

    def test_push_creates_daily_file(self) -> None:
        entry = MemoryEntry(
            date=datetime.date(2026, 3, 22),
            filename="2026-03-22.md",
            content="Today I worked on X.",
        )
        ContextBundleService().push_memory(PROJECT_PREFIX, entry)
        content = self._get(f"{CUSTOMER_PREFIX}/projects/{REPO_NAME}/memory/2026/2026-03-22.md")
        self.assertEqual(content, "Today I worked on X.")

    def test_push_appends_to_existing_file(self) -> None:
        key = f"{CUSTOMER_PREFIX}/projects/{REPO_NAME}/memory/2026/2026-03-22.md"
        self.s3.put_object(Bucket=BUCKET, Key=key, Body=b"Existing content.\n")
        entry = MemoryEntry(
            date=datetime.date(2026, 3, 22),
            filename="2026-03-22.md",
            content="New content.",
        )
        ContextBundleService().push_memory(PROJECT_PREFIX, entry)
        content = self._get(key)
        self.assertEqual(content, "Existing content.\nNew content.")

    def test_push_uses_entry_year_for_path(self) -> None:
        entry = MemoryEntry(
            date=datetime.date(2025, 12, 31),
            filename="2025-12-31.md",
            content="End of year.",
        )
        ContextBundleService().push_memory(PROJECT_PREFIX, entry)
        content = self._get(f"{CUSTOMER_PREFIX}/projects/{REPO_NAME}/memory/2025/2025-12-31.md")
        self.assertEqual(content, "End of year.")


class ContextBundleRoundTripTest(BaseLocalStackTest):
    def _put(self, key: str, content: str) -> None:
        self.s3.put_object(Bucket=BUCKET, Key=key, Body=content.encode("utf-8"))

    def test_pull_push_pull_round_trip(self) -> None:
        self._put(f"{CUSTOMER_PREFIX}/CLAUDE.md", "# Instructions")
        self._put(f"{CUSTOMER_PREFIX}/projects/{REPO_NAME}/README.md", "# Goals")
        self._put(f"{CUSTOMER_PREFIX}/projects/{REPO_NAME}/MEMORY.md", "# Index")

        svc = ContextBundleService()

        bundle1 = svc.pull(PROJECT_PREFIX)
        self.assertEqual(bundle1.claude_md, "# Instructions")
        self.assertEqual(bundle1.project_goals, "# Goals")
        self.assertEqual(bundle1.memory_index, "# Index")
        self.assertEqual(len(bundle1.daily_memories), 0)

        entry = MemoryEntry(
            date=datetime.date(2026, 3, 22),
            filename="2026-03-22.md",
            content="Day 1 notes.",
        )
        svc.push_memory(PROJECT_PREFIX, entry)

        bundle2 = svc.pull(PROJECT_PREFIX)
        self.assertEqual(len(bundle2.daily_memories), 1)
        self.assertEqual(bundle2.daily_memories[0].content, "Day 1 notes.")
        self.assertEqual(bundle2.claude_md, "# Instructions")
        self.assertEqual(bundle2.project_goals, "# Goals")


class PushMemoryNewlineTests(BaseLocalStackTest):
    """push_memory ensures newline separator between existing and new content."""

    def test_newline_added_when_missing(self) -> None:
        key = f"{CUSTOMER_PREFIX}/projects/{REPO_NAME}/memory/2026/2026-03-22.md"
        self.s3.put_object(Bucket=BUCKET, Key=key, Body=b"old content (no trailing newline)")
        entry = MemoryEntry(date=datetime.date(2026, 3, 22), filename="2026-03-22.md", content="## New Entry\n")
        ContextBundleService().push_memory(PROJECT_PREFIX, entry)
        body = self.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode()
        self.assertEqual(body, "old content (no trailing newline)\n## New Entry\n")


class ReadObjectRaisesOnUnexpectedErrorTests(unittest.TestCase):
    """_read_object re-raises non-404 ClientError exceptions."""

    def setUpClass(cls) -> None:
        os.environ.update(_AWS_ENV)

    def test_reraises_non_404_error(self) -> None:
        svc = ContextBundleService()
        s3 = get_s3_client()
        # Bucket does not exist in LocalStack — raises NoSuchBucket (non-404 error code)
        with self.assertRaises(ClientError):
            svc._read_object(s3, "nonexistent-bucket-xyz-does-not-exist", "any/key")
