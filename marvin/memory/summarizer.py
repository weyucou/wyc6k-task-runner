"""Conversation summary pipeline: summarise the oldest message batch when session length exceeds a threshold."""

import datetime
import logging

from marvin.llm import LLMMessage
from marvin.llm.base import BaseLLMClient
from marvin.memory.models import ConversationSummary, Message
from marvin.settings import CONVERSATION_SUMMARY_BATCH_SIZE, CONVERSATION_SUMMARY_THRESHOLD

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPT = (
    "You are a conversation summarizer. Summarize the following conversation messages "
    "into a concise paragraph capturing the key topics, decisions, and context. "
    "Be factual and brief.\n\nMessages:\n{messages}"
)


class ConversationSummarizer:
    def __init__(
        self,
        client: BaseLLMClient,
        threshold: int = CONVERSATION_SUMMARY_THRESHOLD,
        batch_size: int = CONVERSATION_SUMMARY_BATCH_SIZE,
    ) -> None:
        self.client = client
        self.threshold = threshold
        self.batch_size = batch_size

    async def maybe_summarize(
        self,
        session_id: str,
        messages: list[LLMMessage],
    ) -> ConversationSummary | None:
        """Return a ConversationSummary for the oldest batch if len(messages) > threshold, else None."""
        if len(messages) <= self.threshold:
            return None

        batch = messages[: self.batch_size]
        end_index = len(batch) - 1

        messages_text = "\n".join(f"[{m.role}]: {m.content}" for m in batch)
        prompt = _SUMMARIZE_PROMPT.format(messages=messages_text)

        response = await self.client.generate(
            [LLMMessage.user(prompt)],
            temperature=0.3,
        )
        summary_text = response.content

        now = datetime.datetime.now(datetime.UTC)
        preserved = [
            Message(
                role=m.role.value,
                content=m.content,
                timestamp=now,
            )
            for m in batch
        ]

        return ConversationSummary(
            session_id=session_id,
            summary_text=summary_text,
            message_count=len(batch),
            messages=preserved,
            start_index=0,
            end_index=end_index,
        )
