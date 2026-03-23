import json
import uuid
from unittest.mock import MagicMock, patch

from accounts.models import Customer
from django.test import TestCase, override_settings

from tasks.management.commands.run_task_consumer import Command
from tasks.models import TaskEnvelope, TaskStatus

QUEUE_URL = "https://sqs.ap-northeast-1.amazonaws.com/123456789/test-queue"


def _make_sqs_message(body: dict, message_id: str | None = None) -> dict:
    return {
        "MessageId": message_id or str(uuid.uuid4()),
        "ReceiptHandle": "receipt-handle-abc",
        "Body": json.dumps(body),
    }


class RunTaskConsumerTests(TestCase):
    def setUp(self) -> None:
        self.customer = Customer.objects.create(name="SQS Corp")
        self.command = Command()
        self.command.stdout = MagicMock()
        self.command.stderr = MagicMock()
        self.command.style = MagicMock()
        self.command.style.SUCCESS = lambda x: x

    def _make_body(self, **overrides) -> dict:
        body = {
            "customer_id": str(self.customer.id),
            "issue_url": "https://github.com/weyucou/test/issues/5",
            "action": "develop",
            "agent_profile": "coding",
            "context_s3_prefix": "customers/abc/context/",
            "duration_hint_seconds": 3600,
        }
        body.update(overrides)
        return body

    @override_settings(TASK_QUEUE_URL=QUEUE_URL)
    def test_successful_message_creates_envelope_and_deletes(self) -> None:
        mock_sqs = MagicMock()
        msg_id = str(uuid.uuid4())
        message = _make_sqs_message(self._make_body(), message_id=msg_id)
        mock_sqs.receive_message.return_value = {"Messages": [message]}

        self.command._process_message(mock_sqs, QUEUE_URL, message)

        envelope = TaskEnvelope.objects.get(sqs_message_id=msg_id)
        self.assertEqual(envelope.status, TaskStatus.COMPLETE)
        mock_sqs.delete_message.assert_called_once_with(QueueUrl=QUEUE_URL, ReceiptHandle="receipt-handle-abc")

    @override_settings(TASK_QUEUE_URL=QUEUE_URL)
    def test_invalid_json_body_does_not_crash(self) -> None:
        mock_sqs = MagicMock()
        message = {
            "MessageId": str(uuid.uuid4()),
            "ReceiptHandle": "receipt-handle-xyz",
            "Body": "not-json{{",
        }
        self.command._process_message(mock_sqs, QUEUE_URL, message)
        mock_sqs.delete_message.assert_not_called()

    @override_settings(TASK_QUEUE_URL=QUEUE_URL)
    def test_duplicate_sqs_message_id_handled_idempotently(self) -> None:
        mock_sqs = MagicMock()
        msg_id = "dup-msg-001"
        TaskEnvelope.objects.create(
            customer=self.customer,
            issue_url="https://github.com/weyucou/test/issues/1",
            action="develop",
            agent_profile="coding",
            context_s3_prefix="customers/abc/",
            sqs_message_id=msg_id,
        )
        message = _make_sqs_message(self._make_body(), message_id=msg_id)
        self.command._process_message(mock_sqs, QUEUE_URL, message)
        mock_sqs.delete_message.assert_called_once()

    @override_settings(TASK_QUEUE_URL=QUEUE_URL)
    def test_dispatch_failure_does_not_delete_message(self) -> None:
        mock_sqs = MagicMock()
        msg_id = str(uuid.uuid4())
        message = _make_sqs_message(self._make_body(), message_id=msg_id)

        with patch.object(self.command, "_dispatch", return_value=False):
            self.command._process_message(mock_sqs, QUEUE_URL, message)

        mock_sqs.delete_message.assert_not_called()
        self.assertIsNotNone(TaskEnvelope.objects.get(sqs_message_id=msg_id))

    @override_settings(TASK_QUEUE_URL=QUEUE_URL)
    def test_envelope_status_transitions(self) -> None:
        msg_id = str(uuid.uuid4())
        message = _make_sqs_message(self._make_body(), message_id=msg_id)
        mock_sqs = MagicMock()

        self.command._process_message(mock_sqs, QUEUE_URL, message)

        envelope = TaskEnvelope.objects.get(sqs_message_id=msg_id)
        self.assertEqual(envelope.status, TaskStatus.COMPLETE)
        self.assertIsNotNone(envelope.started_at)
        self.assertIsNotNone(envelope.completed_at)

    @override_settings(TASK_QUEUE_URL=QUEUE_URL)
    def test_no_messages_does_not_delete(self) -> None:
        mock_sqs = MagicMock()
        mock_sqs.receive_message.return_value = {"Messages": []}
        self.command._poll_once(mock_sqs, QUEUE_URL)
        mock_sqs.delete_message.assert_not_called()

    def test_sigterm_sets_running_false(self) -> None:
        self.assertTrue(self.command._running)
        self.command._handle_sigterm(15, None)
        self.assertFalse(self.command._running)
