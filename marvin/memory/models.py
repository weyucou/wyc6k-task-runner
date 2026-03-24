"""Pydantic models for the conversation memory system."""

import datetime
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation."""

    role: str
    content: str
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC))


class EmbeddingChunk(BaseModel):
    """A text chunk with its embedding vector for hybrid search retrieval."""

    chunk_id: str
    session_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC))


class ConversationSummary(BaseModel):
    """Summary of a batch of messages from a session.

    The original messages are preserved in ``messages`` to guarantee no data loss.
    The summary text is indexed via an ``EmbeddingChunk`` for hybrid search retrieval.
    """

    summary_id: str
    session_id: str
    summary_text: str
    message_count: int
    messages: list[Message]
    embedding_chunk_id: str
    start_index: int
    end_index: int
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC))
