import logging
from enum import StrEnum

from commons.models import TimestampedModel
from django.conf import settings
from django.db import models

logger = logging.getLogger(__name__)


class ChannelType(StrEnum):
    TELEGRAM = "telegram"
    SLACK = "slack"


class Channel(TimestampedModel):
    """A configured messaging channel (Telegram bot, Slack workspace, etc.)."""

    name = models.CharField(max_length=100)
    channel_type = models.CharField(
        max_length=20,
        choices=[(ct.value, ct.name.title()) for ct in ChannelType],
    )
    is_active = models.BooleanField(default=True)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="channels",
    )

    # Channel-specific configuration stored as JSON
    config = models.JSONField(default=dict, blank=True)

    class Meta:
        unique_together = [("owner", "name")]

    def __str__(self) -> str:
        return f"{self.name} ({self.channel_type})"


class ChannelCredential(TimestampedModel):
    """Encrypted credentials for a channel (API tokens, secrets, etc.)."""

    channel = models.OneToOneField(
        Channel,
        on_delete=models.CASCADE,
        related_name="credential",
    )

    # Telegram: bot_token
    # Slack: bot_token, signing_secret, app_token (for socket mode)
    encrypted_data = models.JSONField(default=dict)

    class Meta:
        verbose_name = "Channel Credential"
        verbose_name_plural = "Channel Credentials"


class Contact(TimestampedModel):
    """A contact/user from a messaging platform."""

    channel = models.ForeignKey(
        Channel,
        on_delete=models.CASCADE,
        related_name="contacts",
    )

    # Platform-specific user ID
    platform_user_id = models.CharField(max_length=255)
    platform_username = models.CharField(max_length=255, blank=True)
    display_name = models.CharField(max_length=255, blank=True)

    # Optional link to Django user if this contact is a known user
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="contacts",
    )

    is_blocked = models.BooleanField(default=False)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        unique_together = [("channel", "platform_user_id")]

    def __str__(self) -> str:
        return self.display_name or self.platform_username or self.platform_user_id


class ChatRoom(TimestampedModel):
    """A chat room/conversation (DM, group, or channel) on a platform."""

    channel = models.ForeignKey(
        Channel,
        on_delete=models.CASCADE,
        related_name="chat_rooms",
    )

    # Platform-specific chat/room ID
    platform_chat_id = models.CharField(max_length=255)
    name = models.CharField(max_length=255, blank=True)

    is_group = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)

    # Agent assigned to handle this room (nullable for routing)
    agent = models.ForeignKey(
        "agents.Agent",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="chat_rooms",
    )

    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        unique_together = [("channel", "platform_chat_id")]

    def __str__(self) -> str:
        return self.name or self.platform_chat_id
