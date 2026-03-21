"""Tests for ProjectContext model and sync_project_context service."""

from unittest.mock import MagicMock, patch

from accounts.models import Customer
from django.test import TestCase

from agents.models import ProjectContext
from agents.services import sync_project_context


def _make_customer(**kwargs) -> Customer:
    defaults = {
        "name": "Test Corp",
        "github_org": "test-corp",
        "s3_context_prefix": "s3://bucket/test-corp/",
    }
    defaults.update(kwargs)
    return Customer.objects.create(**defaults)


def _make_project_context(customer: Customer, **kwargs) -> ProjectContext:
    defaults = {
        "project_id": "PVT_abc123",
        "repo_owner": "test-corp",
        "repo_name": "my-project",
        "s3_prefix": "s3://bucket/test-corp/projects/my-project/",
    }
    defaults.update(kwargs)
    return ProjectContext.objects.create(customer=customer, **defaults)


class ProjectContextModelTest(TestCase):
    def setUp(self) -> None:
        self.customer = _make_customer()

    def test_create(self) -> None:
        ctx = _make_project_context(self.customer)
        self.assertEqual(ctx.project_id, "PVT_abc123")
        self.assertEqual(ctx.repo_owner, "test-corp")
        self.assertEqual(ctx.repo_name, "my-project")
        self.assertEqual(ctx.goals_markdown, "")
        self.assertEqual(ctx.sops_snapshot, {})
        self.assertIsNone(ctx.last_synced)

    def test_str(self) -> None:
        ctx = _make_project_context(self.customer)
        self.assertIn("test-corp", str(ctx))
        self.assertIn("my-project", str(ctx))

    def test_unique_together_customer_project_id(self) -> None:
        _make_project_context(self.customer)
        from django.db import IntegrityError

        with self.assertRaises(IntegrityError):
            _make_project_context(self.customer)

    def test_same_project_id_different_customer(self) -> None:
        other = _make_customer(name="Other Corp", github_org="other-corp", s3_context_prefix="s3://bucket/other/")
        _make_project_context(self.customer)
        ctx2 = _make_project_context(other)
        self.assertEqual(ctx2.project_id, "PVT_abc123")

    def test_timestamps_set_on_create(self) -> None:
        ctx = _make_project_context(self.customer)
        self.assertIsNotNone(ctx.created_datetime)
        self.assertIsNotNone(ctx.updated_datetime)


class SyncProjectContextTest(TestCase):
    def setUp(self) -> None:
        self.customer = _make_customer()
        self.ctx = _make_project_context(self.customer)

    @patch("agents.services._list_s3_sop_files")
    @patch("agents.services._fetch_readme")
    def test_sync_updates_fields(self, mock_readme: MagicMock, mock_sops: MagicMock) -> None:
        mock_readme.return_value = "# My Project\nGoals here."
        mock_sops.return_value = {"sops/overview.md": "Do things safely."}

        sync_project_context(self.ctx.pk)

        self.ctx.refresh_from_db()
        self.assertEqual(self.ctx.goals_markdown, "# My Project\nGoals here.")
        self.assertEqual(self.ctx.sops_snapshot, {"sops/overview.md": "Do things safely."})
        self.assertIsNotNone(self.ctx.last_synced)

    @patch("agents.services._list_s3_sop_files")
    @patch("agents.services._fetch_readme")
    def test_sync_calls_with_correct_args(self, mock_readme: MagicMock, mock_sops: MagicMock) -> None:
        mock_readme.return_value = ""
        mock_sops.return_value = {}

        sync_project_context(self.ctx.pk)

        mock_readme.assert_called_once_with("test-corp", "my-project")
        mock_sops.assert_called_once_with("s3://bucket/test-corp/projects/my-project/")

    @patch("agents.services._list_s3_sop_files")
    @patch("agents.services._fetch_readme")
    def test_sync_empty_readme_and_sops(self, mock_readme: MagicMock, mock_sops: MagicMock) -> None:
        mock_readme.return_value = ""
        mock_sops.return_value = {}

        sync_project_context(self.ctx.pk)

        self.ctx.refresh_from_db()
        self.assertEqual(self.ctx.goals_markdown, "")
        self.assertEqual(self.ctx.sops_snapshot, {})
        self.assertIsNotNone(self.ctx.last_synced)
