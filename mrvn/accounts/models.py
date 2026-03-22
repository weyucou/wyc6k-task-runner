import logging
from enum import StrEnum

from commons.models import TimestampedModel, UUID7Field
from django.contrib.auth.models import AbstractUser
from django.db import models

logger = logging.getLogger(__name__)


class CredentialService(StrEnum):
    github = "github"
    anthropic = "anthropic"
    gemini = "gemini"


class Customer(TimestampedModel):
    """Top-level tenant for per-customer memory isolation."""

    id = UUID7Field(primary_key=True)
    name = models.CharField(max_length=255)
    github_org = models.CharField(max_length=255, unique=True, blank=True, null=True)
    github_token = models.CharField(max_length=512, blank=True, help_text="GitHub API token for this organization")
    # Auto-calculated on save: customers/{id}/
    s3_context_prefix = models.CharField(max_length=512, editable=False, blank=True)
    is_active = models.BooleanField(default=True)

    def save(self, *args, **kwargs):
        self.s3_context_prefix = f"customers/{self.id}/"
        super().save(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.name}({self.id})"


class CustomerCredential(TimestampedModel):
    """Maps a customer's service to its secret ARN in AWS Secrets Manager."""

    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="credentials")
    service = models.CharField(max_length=64, choices=[(s.value, s.name) for s in CredentialService])
    secret_arn = models.CharField(max_length=512)

    class Meta:
        unique_together = [("customer", "service")]

    def __str__(self) -> str:
        return f"{self.customer} / {self.service}"


class CustomUser(AbstractUser):
    @property
    def email_domain(self):
        domain = self.email.split("@")[-1]  # NAME@DOMAIN.COM -> [ 'NAME', 'DOMAIN.COM']
        return domain

    @property
    def display_name(self):
        return f"{self.last_name}, {self.first_name} ({self.username})"
