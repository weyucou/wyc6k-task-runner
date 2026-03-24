"""Unit tests for S3MemoryWriteTool using moto to mock S3."""

import datetime
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import boto3
from moto import mock_aws

from agents.tools.base import ToolStatus
from agents.tools.s3_memory import S3MemoryWriteTool

BUCKET = "test-bucket"
CUSTOMER_PREFIX = "customers/cust-001"
REPO_NAME = "my-repo"
S3_PREFIX = f"s3://{BUCKET}/{CUSTOMER_PREFIX}/projects/{REPO_NAME}"

_MOTO_ENV = {
    "AWS_ACCESS_KEY_ID": "test",
    "AWS_SECRET_ACCESS_KEY": "test",
    "AWS_DEFAULT_REGION": "us-east-1",
}


class S3MemoryWriteToolTests(IsolatedAsyncioTestCase):
    """Tests for S3MemoryWriteTool.execute() — uses moto to mock S3."""

    async def asyncSetUp(self) -> None:
        self._mock = mock_aws()
        self._mock.start()
        self._env_patcher = patch.dict("os.environ", _MOTO_ENV)
        self._env_patcher.start()
        self._settings_patcher = patch("django.conf.settings.S3_ENDPOINT_URL", None)
        self._settings_patcher.start()
        self.s3 = boto3.client("s3", region_name="us-east-1")
        self.s3.create_bucket(Bucket=BUCKET)
        self.tool = S3MemoryWriteTool(s3_prefix=S3_PREFIX)

    async def asyncTearDown(self) -> None:
        self._settings_patcher.stop()
        self._env_patcher.stop()
        self._mock.stop()

    async def test_execute_creates_daily_file(self) -> None:
        today = datetime.datetime.now(tz=datetime.UTC).date()
        result = await self.tool.execute(
            section_header="## Actions",
            content="- Updated issue #9\n",
        )

        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertIn(today.isoformat(), result.output)
        self.assertEqual(result.data["date"], today.isoformat())

    async def test_execute_appends_section_header_and_content(self) -> None:
        today = datetime.datetime.now(tz=datetime.UTC).date()
        await self.tool.execute(
            section_header="## 2026-03-18 14:30:00 Actions",
            content="- Did something\n",
        )

        key = f"{CUSTOMER_PREFIX}/projects/{REPO_NAME}/memory/{today.year}/{today.isoformat()}.md"
        body = self.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode()

        self.assertIn("## 2026-03-18 14:30:00 Actions", body)
        self.assertIn("- Did something", body)

    async def test_execute_appends_to_existing_file(self) -> None:
        today = datetime.datetime.now(tz=datetime.UTC).date()
        key = f"{CUSTOMER_PREFIX}/projects/{REPO_NAME}/memory/{today.year}/{today.isoformat()}.md"
        self.s3.put_object(Bucket=BUCKET, Key=key, Body=b"## Earlier Entry\n- old note\n")

        await self.tool.execute(section_header="## Later Entry", content="- new note\n")

        body = self.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode()
        self.assertIn("## Earlier Entry", body)
        self.assertIn("## Later Entry", body)
        self.assertIn("- new note", body)

    async def test_execute_returns_error_when_no_prefix(self) -> None:
        tool = S3MemoryWriteTool(s3_prefix="")
        result = await tool.execute(section_header="## Actions", content="content\n")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("s3_prefix", result.error)

    async def test_execute_data_contains_filename_and_prefix(self) -> None:
        today = datetime.datetime.now(tz=datetime.UTC).date()
        result = await self.tool.execute(section_header="## Header", content="body\n")
        self.assertEqual(result.data["s3_prefix"], S3_PREFIX)
        self.assertEqual(result.data["filename"], f"{today.isoformat()}.md")


class S3MemoryWriteToolProfileTests(unittest.TestCase):
    """Tests that S3MemoryWriteTool appears in the expected tool profiles."""

    def test_tool_name(self) -> None:
        self.assertEqual(S3MemoryWriteTool.name, "s3_memory_write")

    def test_tool_registered_in_coding_profile(self) -> None:
        from agents.models import Agent, ToolProfile  # noqa: PLC0415

        agent = Agent.__new__(Agent)
        agent.tool_profile = ToolProfile.CODING.value
        agent.tools_allow = []
        agent.tools_deny = []

        all_tools = ["s3_memory_write", "read", "write", "exec"]
        allowed = agent.get_allowed_tools(all_tools)
        self.assertIn("s3_memory_write", allowed)

    def test_tool_registered_in_full_profile(self) -> None:
        from agents.models import Agent, ToolProfile  # noqa: PLC0415

        agent = Agent.__new__(Agent)
        agent.tool_profile = ToolProfile.FULL.value
        agent.tools_allow = []
        agent.tools_deny = []

        all_tools = ["s3_memory_write", "read", "write", "exec"]
        allowed = agent.get_allowed_tools(all_tools)
        self.assertIn("s3_memory_write", allowed)
