import logging

from django.db.models.signals import post_save
from django.dispatch import receiver

from memory.models import Message, MessageRole
from memory.search import memory_search_service

logger = logging.getLogger(__name__)

INDEXED_ROLES = {MessageRole.USER, MessageRole.ASSISTANT}


@receiver(post_save, sender=Message)
def handle_message_post_save(_sender: type[Message], instance: Message, created: bool, **_kwargs: object) -> None:
    """Index new messages for hybrid memory search on creation."""
    if not created:
        return
    if instance.role not in INDEXED_ROLES:
        return
    agent_id = instance.session.agent_id
    if agent_id is None:
        logger.debug("Skipping indexing for message %s: session has no agent", instance.pk)
        return
    memory_search_service.index_message(instance, agent_id)
