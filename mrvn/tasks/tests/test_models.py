import uuid

from accounts.models import Customer
from django.db import IntegrityError
from django.test import TestCase

from tasks.models import TaskEnvelope, TaskStatus


class TaskEnvelopeModelTests(TestCase):
    def setUp(self) -> None:
        self.customer = Customer.objects.create(name="Test Corp")

    def _make_envelope(self, sqs_message_id: str = "msg-001", **kwargs) -> TaskEnvelope:
        defaults = {
            "customer": self.customer,
            "issue_url": "https://github.com/weyucou/test/issues/1",
            "action": "develop",
            "agent_profile": "coding",
            "context_s3_prefix": "customers/123/context/",
            "sqs_message_id": sqs_message_id,
        }
        defaults.update(kwargs)
        return TaskEnvelope.objects.create(**defaults)

    def test_create(self) -> None:
        envelope = self._make_envelope()
        self.assertIsNotNone(envelope.pk)
        self.assertIsInstance(envelope.pk, uuid.UUID)

    def test_default_status_is_queued(self) -> None:
        envelope = self._make_envelope()
        self.assertEqual(envelope.status, TaskStatus.QUEUED)

    def test_default_duration_hint(self) -> None:
        envelope = self._make_envelope()
        self.assertEqual(envelope.duration_hint_seconds, 3600)

    def test_str(self) -> None:
        envelope = self._make_envelope()
        self.assertIn("develop", str(envelope))

    def test_sqs_message_id_unique(self) -> None:
        self._make_envelope(sqs_message_id="unique-001")
        with self.assertRaises(IntegrityError):
            self._make_envelope(sqs_message_id="unique-001")

    def test_project_context_optional(self) -> None:
        envelope = self._make_envelope()
        self.assertIsNone(envelope.project_context)

    def test_timestamps(self) -> None:
        envelope = self._make_envelope()
        self.assertIsNotNone(envelope.created_datetime)
        self.assertIsNotNone(envelope.updated_datetime)
