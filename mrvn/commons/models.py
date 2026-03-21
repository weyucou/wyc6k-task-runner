from django.conf import settings
from django.db import models

from commons.functions import uuidv7


class UUID7Field(models.UUIDField):
    """UUIDField that defaults to a time-ordered UUIDv7."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("default", uuidv7)
        kwargs.setdefault("editable", False)
        super().__init__(*args, **kwargs)


class TimestampedModel(models.Model):
    created_datetime = models.DateTimeField(auto_now_add=True, editable=False)
    updated_datetime = models.DateTimeField(auto_now=True, editable=False)

    class Meta:
        abstract = True


class UserTimestampedModel(TimestampedModel):
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        editable=False,
        related_name="%(app_label)s_%(class)s_created_by",
    )
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        editable=False,
        related_name="%(app_label)s_%(class)s_updated_by",
    )

    class Meta:
        abstract = True
