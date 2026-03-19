import logging

from commons.models import TimestampedModel
from django.contrib.auth.models import AbstractUser
from django.db import models

logger = logging.getLogger(__name__)


class Customer(TimestampedModel):
    """Top-level tenant for per-customer memory isolation."""

    name = models.CharField(max_length=255)
    github_org = models.CharField(max_length=255, unique=True)
    # Key prefix in S3, e.g. cust_abc/ (bucket configured separately)
    s3_context_prefix = models.CharField(max_length=512)
    is_active = models.BooleanField(default=True)

    def __str__(self) -> str:
        return self.name


class CustomUser(AbstractUser):
    @property
    def email_domain(self):
        domain = self.email.split("@")[-1]  # NAME@DOMAIN.COM -> [ 'NAME', 'DOMAIN.COM']
        return domain

    @property
    def display_name(self):
        return f"{self.last_name}, {self.first_name} ({self.username})"
