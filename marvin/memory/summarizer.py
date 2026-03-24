"""Conversation summary pipeline: summarise old messages and index to EmbeddingChunk."""

import logging
import os
import uuid

from marvin.llm import LLMMessage
from marvin.llm.factory import create_client_from_agent_config
from marvin.memory.models import ConversationSummary, EmbeddingChunk, Message
from marvin.models import AgentConfig

logger = logging.getLogger(__name__)

SUMMARY_THRESHOLD = int(os.getenv("CONVERSATION_SUMMARY_THRESHOLD", "50"))
SUMMARY_BATCH_SIZE = int(os.getenv("CONVERSATION_SUMMARY_BATCH_SIZE", "20"))
SUMMARY_EMBED_MODEL = os.getenv("CONVERSATION_SUMMARY_EMBED_MODEL", "all-MiniLM-L6-v2")

_SUMMARIZE_SYSTEM_PROMPT = (
    "You are a concise assistant. Summarize the following conversation messages into a brief but complete summary "
    "that captures all key facts, decisions, and outcomes. Be factual and preserve important details."
)
_SUMMARIZE_USER_TEMPLATE = "Summarize the following conversation messages:\n\n{messages}"


def _format_messages(messages: list[Message]) -> str:
    return "\n".join(f"[{m.role.upper()}]: {m.content}" for m in messages)


def _compute_embedding(text: str, model_name: str) -> list[float]:
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer(model_name)
    return model.encode(text).tolist()


class ConversationSummarizer:
    """Summarises the oldest message batch when session length exceeds a threshold.

    Configuration (env vars with defaults):
    - CONVERSATION_SUMMARY_THRESHOLD (default 50): minimum message count before summarising
    - CONVERSATION_SUMMARY_BATCH_SIZE (default 20): number of oldest messages to summarise
    - CONVERSATION_SUMMARY_EMBED_MODEL (default "all-MiniLM-L6-v2"): sentence-transformers model
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        *,
        threshold: int | None = None,
        batch_size: int | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.agent_config = agent_config
        self.threshold = threshold if threshold is not None else SUMMARY_THRESHOLD
        self.batch_size = batch_size if batch_size is not None else SUMMARY_BATCH_SIZE
        self.embed_model = embed_model or SUMMARY_EMBED_MODEL

    async def maybe_summarize(
        self,
        session_id: str,
        messages: list[LLMMessage],
    ) -> tuple[ConversationSummary, EmbeddingChunk] | None:
        """Create a summary if the message count exceeds the threshold.

        Returns:
            A (ConversationSummary, EmbeddingChunk) tuple when summarisation occurs,
            or ``None`` when the threshold is not exceeded.
        """
        if len(messages) <= self.threshold:
            return None

        batch = messages[: self.batch_size]
        memory_messages = [Message(role=m.role.value, content=m.content) for m in batch]

        summary_text = await self._call_llm_for_summary(memory_messages)

        chunk_id = str(uuid.uuid4())
        summary_id = str(uuid.uuid4())

        embedding = _compute_embedding(summary_text, self.embed_model)

        chunk = EmbeddingChunk(
            chunk_id=chunk_id,
            session_id=session_id,
            text=summary_text,
            embedding=embedding,
        )
        summary = ConversationSummary(
            summary_id=summary_id,
            session_id=session_id,
            summary_text=summary_text,
            message_count=len(batch),
            messages=memory_messages,
            embedding_chunk_id=chunk_id,
            start_index=0,
            end_index=len(batch),
        )

        logger.info(
            "Created ConversationSummary %s for session %s (%d messages summarised)",
            summary_id,
            session_id,
            len(batch),
        )
        return summary, chunk

    async def _call_llm_for_summary(self, messages: list[Message]) -> str:
        client = create_client_from_agent_config(self.agent_config)
        formatted = _format_messages(messages)
        user_content = _SUMMARIZE_USER_TEMPLATE.format(messages=formatted)
        response = await client.generate(
            [LLMMessage.user(user_content)],
            system_prompt=_SUMMARIZE_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.content
