"""Tests for CustomerContextBundle and ContextBundleService."""

import datetime
import unittest
from typing import Any

import boto3
from moto import mock_aws

from agents.context import ContextBundleService, CustomerContextBundle, MemoryEntry

BUCKET = "weyucou-agent-contexts"
CUSTOMER_PREFIX = "customers/cust-001"
REPO_NAME = "my-repo"
PROJECT_PREFIX = f"s3://{BUCKET}/{CUSTOMER_PREFIX}/projects/{REPO_NAME}"


def _setup_bucket(s3: Any) -> None:
    s3.create_bucket(Bucket=BUCKET)


@mock_aws
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


@mock_aws
class ContextBundleServicePullTest(unittest.TestCase):
    def setUp(self) -> None:
        self.s3 = boto3.client("s3", region_name="us-east-1")
        _setup_bucket(self.s3)

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


@mock_aws
class ContextBundleServicePushTest(unittest.TestCase):
    def setUp(self) -> None:
        self.s3 = boto3.client("s3", region_name="us-east-1")
        _setup_bucket(self.s3)

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


@mock_aws
class ContextBundleRoundTripTest(unittest.TestCase):
    def setUp(self) -> None:
        self.s3 = boto3.client("s3", region_name="us-east-1")
        _setup_bucket(self.s3)

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
