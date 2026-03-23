import json
import logging
import signal
import time
from typing import TYPE_CHECKING, Any

import boto3
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import IntegrityError
from django.utils import timezone

from tasks.models import TaskEnvelope, TaskStatus

if TYPE_CHECKING:
    from argparse import ArgumentParser

logger = logging.getLogger(__name__)

WAIT_TIME_SECONDS = 20  # SQS long-poll max
MAX_MESSAGES = 1  # Process one at a time for simplicity


class Command(BaseCommand):
    help = "Poll SQS task queue and dispatch received TaskEnvelopes"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._running = True

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--queue-url",
            default="",
            help="SQS queue URL (overrides TASK_QUEUE_URL setting)",
        )

    def handle(self, *args, **options) -> None:
        queue_url = options["queue_url"] or settings.TASK_QUEUE_URL
        if not queue_url:
            raise CommandError("TASK_QUEUE_URL is not configured. Set the TASK_QUEUE_URL environment variable.")

        signal.signal(signal.SIGTERM, self._handle_sigterm)

        sqs = boto3.client("sqs", region_name=settings.AWS_REGION)
        self.stdout.write(f"Starting SQS task consumer on queue: {queue_url}")

        while self._running:
            try:
                self._poll_once(sqs, queue_url)
            except Exception:
                logger.exception("Unexpected error in consumer loop")
                if self._running:
                    time.sleep(5)

        self.stdout.write(self.style.SUCCESS("Task consumer stopped gracefully."))

    def _poll_once(self, sqs: Any, queue_url: str) -> None:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=MAX_MESSAGES,
            WaitTimeSeconds=WAIT_TIME_SECONDS,
        )
        messages = response.get("Messages", [])
        if not messages:
            return

        for message in messages:
            if not self._running:
                break
            self._process_message(sqs, queue_url, message)

    def _process_message(self, sqs: Any, queue_url: str, message: dict) -> None:
        sqs_message_id = message["MessageId"]
        receipt_handle = message["ReceiptHandle"]

        try:
            body = json.loads(message["Body"])
        except json.JSONDecodeError:
            logger.exception("Failed to parse SQS message body (id=%s)", sqs_message_id)
            return

        envelope = self._create_envelope(sqs_message_id, body)
        if envelope is None:
            logger.info("Duplicate sqs_message_id=%s — deleting from queue", sqs_message_id)
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            return

        dispatched = self._dispatch(envelope)
        if dispatched:
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
            logger.info("Dispatched and deleted message sqs_message_id=%s", sqs_message_id)
        else:
            logger.warning(
                "Dispatch failed for sqs_message_id=%s — message NOT deleted; will return to queue",
                sqs_message_id,
            )

    def _create_envelope(self, sqs_message_id: str, body: dict) -> TaskEnvelope | None:
        from accounts.models import Customer  # noqa: PLC0415
        from agents.models import ProjectContext  # noqa: PLC0415

        try:
            customer = Customer.objects.get(id=body["customer_id"])
        except (Customer.DoesNotExist, KeyError):
            logger.exception("Cannot resolve customer from message body")
            return None

        project_context = None
        if project_context_id := body.get("project_context_id"):
            try:
                project_context = ProjectContext.objects.get(id=project_context_id)
            except ProjectContext.DoesNotExist:
                logger.warning("ProjectContext id=%s not found — proceeding without it", project_context_id)

        try:
            envelope = TaskEnvelope.objects.create(
                customer=customer,
                project_context=project_context,
                issue_url=body.get("issue_url", ""),
                action=body.get("action", ""),
                agent_profile=body.get("agent_profile", ""),
                context_s3_prefix=body.get("context_s3_prefix", ""),
                duration_hint_seconds=body.get("duration_hint_seconds", 3600),
                sqs_message_id=sqs_message_id,
            )
        except IntegrityError:
            logger.warning("IntegrityError: sqs_message_id=%s already exists", sqs_message_id)
            return None

        return envelope

    def _dispatch(self, envelope: TaskEnvelope) -> bool:
        envelope.started_at = timezone.now()
        envelope.status = TaskStatus.DISPATCHED
        envelope.save(update_fields=["started_at", "status", "updated_datetime"])

        try:
            # TODO: replace stub with AdapterFactory.select(envelope).dispatch(envelope) after #10 merges
            result = None
        except Exception as exc:
            logger.exception("Dispatch failed for envelope id=%s", envelope.id)
            envelope.status = TaskStatus.FAILED
            envelope.error = str(exc)
            envelope.save(update_fields=["status", "error", "updated_datetime"])
            return False
        else:
            envelope.status = TaskStatus.COMPLETE
            envelope.completed_at = timezone.now()
            envelope.dispatch_result = result
            envelope.save(update_fields=["status", "completed_at", "dispatch_result", "updated_datetime"])
            return True

    def _handle_sigterm(self, signum: int, frame: Any) -> None:
        self.stdout.write("Received SIGTERM — stopping after current message...")
        self._running = False
