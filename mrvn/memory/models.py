import logging
from enum import StrEnum

from commons.models import TimestampedModel
from django.db import models
from pgvector.django import HnswIndex, VectorField
from psqlextra.models import PostgresPartitionedModel
from psqlextra.types import PostgresPartitioningMethod

logger = logging.getLogger(__name__)

# Default embedding dimensions (all-MiniLM-L6-v2)
DEFAULT_EMBEDDING_DIMENSIONS = 384

CONTENT_PREVIEW_LENGTH = 50


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Session(TimestampedModel):
    """A conversation session tied to a chat room."""

    chat_room = models.ForeignKey(
        "channels.ChatRoom",
        on_delete=models.CASCADE,
        related_name="sessions",
    )

    # Session can be linked to a specific contact
    contact = models.ForeignKey(
        "channels.Contact",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="sessions",
    )

    # Agent handling this session
    agent = models.ForeignKey(
        "agents.Agent",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="sessions",
    )

    is_active = models.BooleanField(default=True)

    # Session metadata (context window info, token counts, etc.)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_datetime"]

    def __str__(self) -> str:
        return f"Session {self.pk} - {self.chat_room}"


class Message(TimestampedModel):
    """A single message in a conversation."""

    session = models.ForeignKey(
        Session,
        on_delete=models.CASCADE,
        related_name="messages",
    )

    role = models.CharField(
        max_length=20,
        choices=[(r.value, r.name.title()) for r in MessageRole],
    )
    content = models.TextField()

    # Platform message ID for deduplication/tracking
    platform_message_id = models.CharField(max_length=255, blank=True)

    # For tool messages
    tool_call_id = models.CharField(max_length=255, blank=True)
    tool_name = models.CharField(max_length=255, blank=True)

    # Token usage tracking
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)

    # Store raw API response/request for debugging
    raw_data = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["created_datetime"]

    def __str__(self) -> str:
        if len(self.content) > CONTENT_PREVIEW_LENGTH:
            preview = self.content[:CONTENT_PREVIEW_LENGTH] + "..."
        else:
            preview = self.content
        return f"[{self.role}] {preview}"


class ConversationSummary(TimestampedModel):
    """Summarized context from older messages to maintain long-term memory."""

    session = models.ForeignKey(
        Session,
        on_delete=models.CASCADE,
        related_name="summaries",
    )

    summary = models.TextField()
    messages_summarized = models.IntegerField(default=0)

    # Range of messages this summary covers
    first_message = models.ForeignKey(
        Message,
        on_delete=models.SET_NULL,
        null=True,
        related_name="+",
    )
    last_message = models.ForeignKey(
        Message,
        on_delete=models.SET_NULL,
        null=True,
        related_name="+",
    )

    class Meta:
        ordering = ["-created_datetime"]
        verbose_name_plural = "Conversation Summaries"

    def __str__(self) -> str:
        return f"Summary for Session {self.session_id} ({self.messages_summarized} messages)"


class ChunkSource(StrEnum):
    """Source type for embedding chunks."""

    MESSAGE = "message"
    SUMMARY = "summary"
    FILE = "file"


class EmbeddingChunk(PostgresPartitionedModel):
    """Vector embeddings for memory search, partitioned by agent.

    Uses pgvector for efficient similarity search and psqlextra
    for table partitioning on agent_id.
    """

    class PartitioningMeta:
        method = PostgresPartitioningMethod.LIST
        key = ["agent_id"]

    id = models.BigAutoField(primary_key=True)

    # Partition key - required for list partitioning
    agent = models.ForeignKey(
        "agents.Agent",
        on_delete=models.CASCADE,
        related_name="embedding_chunks",
    )

    # Source reference (message, summary, or file)
    source = models.CharField(
        max_length=20,
        choices=[(s.value, s.name.title()) for s in ChunkSource],
        default=ChunkSource.MESSAGE.value,
    )
    source_id = models.BigIntegerField(
        help_text="ID of the source message, summary, or file",
    )

    # Chunk content and position
    text = models.TextField()
    start_line = models.IntegerField(default=0)
    end_line = models.IntegerField(default=0)

    # Embedding vector (pgvector)
    embedding = VectorField(
        dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
        help_text="Vector embedding for similarity search",
    )

    # Embedding metadata
    embedding_model = models.CharField(
        max_length=100,
        default="all-MiniLM-L6-v2",
        help_text="Model used to generate embedding",
    )
    content_hash = models.CharField(
        max_length=64,
        db_index=True,
        help_text="SHA256 hash for deduplication",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            # HNSW index for fast approximate nearest neighbor search
            # m=24: More connections per layer improves recall (default=16)
            # ef_construction=128: Better graph quality during build (default=64)
            # See: https://github.com/pgvector/pgvector#hnsw
            HnswIndex(
                name="embedding_chunk_hnsw_idx",
                fields=["embedding"],
                m=24,
                ef_construction=128,
                opclasses=["vector_cosine_ops"],
            ),
            models.Index(fields=["agent", "source", "source_id"]),
            models.Index(fields=["agent", "content_hash"]),
        ]

    def __str__(self) -> str:
        return f"Chunk {self.id} [{self.source}:{self.source_id}]"


class EmbeddingCache(models.Model):
    """Cache for pre-computed embeddings to avoid recomputation.

    Keyed by (model, content_hash) for reuse across agents.
    """

    embedding_model = models.CharField(max_length=100)
    content_hash = models.CharField(max_length=64)
    embedding = VectorField(dimensions=DEFAULT_EMBEDDING_DIMENSIONS)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("embedding_model", "content_hash")]
        indexes = [
            models.Index(fields=["embedding_model", "content_hash"]),
        ]

    def __str__(self) -> str:
        return f"Cache [{self.embedding_model}] {self.content_hash[:16]}..."
