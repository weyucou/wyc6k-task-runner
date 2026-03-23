import logging

from commons.models import TimestampedModel, UUID7Field
from django.db import models

logger = logging.getLogger(__name__)


class TaskStatus(models.TextChoices):
    QUEUED = "queued", "Queued"
    DISPATCHED = "dispatched", "Dispatched"
    COMPLETE = "complete", "Complete"
    FAILED = "failed", "Failed"


class TaskEnvelope(TimestampedModel):
    id = UUID7Field(primary_key=True)
    customer = models.ForeignKey(
        "accounts.Customer",
        on_delete=models.CASCADE,
        related_name="task_envelopes",
    )
    project_context = models.ForeignKey(
        "agents.ProjectContext",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="task_envelopes",
    )
    issue_url = models.CharField(max_length=512)
    action = models.CharField(max_length=64)
    agent_profile = models.CharField(max_length=64)
    context_s3_prefix = models.CharField(max_length=512)
    duration_hint_seconds = models.IntegerField(default=3600)
    status = models.CharField(
        max_length=32,
        choices=TaskStatus.choices,
        default=TaskStatus.QUEUED,
    )
    sqs_message_id = models.CharField(max_length=255, unique=True)
    dispatch_result = models.JSONField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_datetime"]

    def __str__(self) -> str:
        return f"TaskEnvelope({self.action}:{self.issue_url})"
