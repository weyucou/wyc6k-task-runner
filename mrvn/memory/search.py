"""Memory search service with pgvector + text hybrid search.

Implements OpenClaw-inspired memory search with:
- Vector embeddings stored in PostgreSQL with pgvector
- Text search for keyword matching
- Hybrid scoring with configurable weights
- Table partitioning by (customer_id, agent_id) for performance
"""

import hashlib
import logging
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from django.db.models import Q
from pydantic import BaseModel, Field

from memory.models import (
    SENTINEL_CUSTOMER_ID,
    ChunkSource,
    ConversationSummary,
    EmbeddingCache,
    EmbeddingChunk,
    Message,
)

if TYPE_CHECKING:
    from memory.models import Session

logger = logging.getLogger(__name__)


class MemorySearchConfig(BaseModel):
    """Configuration for memory search."""

    enabled: bool = True
    session_memory: bool = True  # Include session history
    chunk_size: int = 400  # Tokens per chunk
    chunk_overlap: int = 80
    max_results: int = 6
    min_score: float = 0.35
    hybrid_weights: dict[str, float] = Field(default_factory=lambda: {"vector": 0.7, "text": 0.3})
    embedding_model: str = "all-MiniLM-L6-v2"
    # HNSW ef_search: higher = better recall, slightly slower (default=40)
    # Recommended: 100-200 for production RAG systems
    ef_search: int = 100

    model_config = {"frozen": False}


class MemorySearchResult(BaseModel):
    """A single search result."""

    content: str
    score: float
    source: Literal["message", "summary"]
    message_id: int | None = None
    summary_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchService:
    """Service for searching agent memory using pgvector + text hybrid search."""

    def __init__(self, config: MemorySearchConfig | None = None) -> None:
        """Initialize the search service.

        Args:
            config: Optional search configuration.
        """
        self.config = config or MemorySearchConfig()
        self._embedder = None

    def _get_embedder(self):  # noqa: ANN202
        """Lazy load the sentence transformer model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415

                self._embedder = SentenceTransformer(self.config.embedding_model)
            except ImportError:
                logger.warning("sentence-transformers not installed, vector search disabled")
                self._embedder = False  # Mark as unavailable
        return self._embedder if self._embedder else None

    def get_embedding(self, text: str) -> list[float] | None:
        """Get embedding vector for text, using cache if available.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding, or None if unavailable.
        """
        embedder = self._get_embedder()
        if not embedder:
            return None

        # Check cache first
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        cached = EmbeddingCache.objects.filter(
            embedding_model=self.config.embedding_model,
            content_hash=content_hash,
        ).first()

        if cached:
            return list(cached.embedding)

        # Generate new embedding
        embedding = embedder.encode(text, convert_to_numpy=True)
        embedding_list = embedding.tolist()

        # Cache it
        EmbeddingCache.objects.create(
            embedding_model=self.config.embedding_model,
            content_hash=content_hash,
            embedding=embedding_list,
        )

        return embedding_list

    def content_hash(self, text: str) -> str:
        """Generate SHA256 hash for content deduplication."""
        return hashlib.sha256(text.encode()).hexdigest()

    def index_message(self, message: Message, agent_id: int, customer_id: UUID | None = None) -> EmbeddingChunk | None:
        """Index a message for vector search.

        Args:
            message: Message to index.
            agent_id: Agent ID for partitioning.
            customer_id: Customer ID for tenant isolation. Defaults to SENTINEL_CUSTOMER_ID.

        Returns:
            Created EmbeddingChunk or None if embedding failed.
        """
        embedding = self.get_embedding(message.content)
        if not embedding:
            return None

        effective_customer_id = customer_id or SENTINEL_CUSTOMER_ID
        content_hash = self.content_hash(message.content)

        # Check if already indexed
        existing = EmbeddingChunk.objects.filter(
            agent_id=agent_id,
            customer_id=effective_customer_id,
            source=ChunkSource.MESSAGE.value,
            source_id=message.id,
        ).first()

        if existing:
            # Update if content changed
            if existing.content_hash != content_hash:
                existing.text = message.content
                existing.embedding = embedding
                existing.content_hash = content_hash
                existing.save()
            return existing

        return EmbeddingChunk.objects.create(
            agent_id=agent_id,
            customer_id=effective_customer_id,
            source=ChunkSource.MESSAGE.value,
            source_id=message.id,
            text=message.content,
            embedding=embedding,
            embedding_model=self.config.embedding_model,
            content_hash=content_hash,
        )

    def index_summary(self, summary: ConversationSummary, agent_id: int, customer_id: UUID | None = None) -> EmbeddingChunk | None:
        """Index a conversation summary for vector search."""
        embedding = self.get_embedding(summary.summary)
        if not embedding:
            return None

        effective_customer_id = customer_id or SENTINEL_CUSTOMER_ID
        content_hash = self.content_hash(summary.summary)

        existing = EmbeddingChunk.objects.filter(
            agent_id=agent_id,
            customer_id=effective_customer_id,
            source=ChunkSource.SUMMARY.value,
            source_id=summary.id,
        ).first()

        if existing:
            if existing.content_hash != content_hash:
                existing.text = summary.summary
                existing.embedding = embedding
                existing.content_hash = content_hash
                existing.save()
            return existing

        return EmbeddingChunk.objects.create(
            agent_id=agent_id,
            customer_id=effective_customer_id,
            source=ChunkSource.SUMMARY.value,
            source_id=summary.id,
            text=summary.summary,
            embedding=embedding,
            embedding_model=self.config.embedding_model,
            content_hash=content_hash,
        )

    def text_search(
        self,
        query: str,
        session: Session | None = None,
        agent_id: int | None = None,
        customer_id: UUID | None = None,
    ) -> list[MemorySearchResult]:
        """Perform text-based search on messages.

        Args:
            query: Search query.
            session: Optional session to limit search.
            agent_id: Optional agent ID to limit search.
            customer_id: Optional customer ID to scope results to a tenant.

        Returns:
            List of search results.
        """
        results: list[MemorySearchResult] = []
        query_lower = query.lower()
        words = query_lower.split()

        # When customer_id is set, restrict messages/summaries to those indexed
        # under that customer via EmbeddingChunk cross-reference subquery.
        # (Message has no direct customer FK, so we scope through EmbeddingChunk.)
        if customer_id is not None:
            indexed_message_ids = EmbeddingChunk.objects.filter(
                customer_id=customer_id,
                source=ChunkSource.MESSAGE.value,
            ).values_list("source_id", flat=True)
            indexed_summary_ids = EmbeddingChunk.objects.filter(
                customer_id=customer_id,
                source=ChunkSource.SUMMARY.value,
            ).values_list("source_id", flat=True)
        else:
            indexed_message_ids = None
            indexed_summary_ids = None

        # Search messages
        message_qs = Message.objects.all()
        if session:
            message_qs = message_qs.filter(session=session)
        elif agent_id:
            message_qs = message_qs.filter(session__agent_id=agent_id)
        if indexed_message_ids is not None:
            message_qs = message_qs.filter(id__in=indexed_message_ids)

        q_filter = Q()
        for word in words:
            q_filter |= Q(content__icontains=word)

        messages = message_qs.filter(q_filter).order_by("-created_datetime")[: self.config.max_results * 2]

        for msg in messages:
            content_lower = msg.content.lower()
            matches = sum(1 for word in words if word in content_lower)
            score = matches / len(words) if words else 0

            if score >= self.config.min_score:
                results.append(
                    MemorySearchResult(
                        content=msg.content,
                        score=score,
                        source="message",
                        message_id=msg.id,
                        metadata={
                            "role": msg.role,
                            "created": msg.created_datetime.isoformat() if msg.created_datetime else None,
                        },
                    )
                )

        # Search summaries
        summary_qs = ConversationSummary.objects.all()
        if session:
            summary_qs = summary_qs.filter(session=session)
        elif agent_id:
            summary_qs = summary_qs.filter(session__agent_id=agent_id)
        if indexed_summary_ids is not None:
            summary_qs = summary_qs.filter(id__in=indexed_summary_ids)

        summaries = summary_qs.filter(summary__icontains=query)[: self.config.max_results]
        for summary in summaries:
            content_lower = summary.summary.lower()
            matches = sum(1 for word in words if word in content_lower)
            score = matches / len(words) if words else 0

            if score >= self.config.min_score:
                results.append(
                    MemorySearchResult(
                        content=summary.summary,
                        score=score,
                        source="summary",
                        summary_id=summary.id,
                        metadata={
                            "messages_summarized": summary.messages_summarized,
                            "created": summary.created_datetime.isoformat() if summary.created_datetime else None,
                        },
                    )
                )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[: self.config.max_results]

    def vector_search(
        self,
        query: str,
        session: Session | None = None,
        agent_id: int | None = None,
        customer_id: UUID | None = None,
    ) -> list[MemorySearchResult]:
        """Perform vector-based semantic search using pgvector.

        Args:
            query: Search query.
            session: Optional session to limit search.
            agent_id: Optional agent ID to limit search.
            customer_id: Optional customer ID to scope results to a tenant.

        Returns:
            List of search results sorted by similarity.
        """
        from django.db import connection  # noqa: PLC0415
        from pgvector.django import CosineDistance  # noqa: PLC0415

        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []

        results: list[MemorySearchResult] = []

        # Set HNSW ef_search for better recall at query time
        # Higher values improve recall with sublinear speed decrease
        with connection.cursor() as cursor:
            cursor.execute(f"SET hnsw.ef_search = {self.config.ef_search}")

        # Query EmbeddingChunk with pgvector cosine distance
        chunk_qs = EmbeddingChunk.objects.all()

        if agent_id:
            chunk_qs = chunk_qs.filter(agent_id=agent_id)
        if customer_id is not None:
            chunk_qs = chunk_qs.filter(customer_id=customer_id)

        # Use pgvector's cosine distance operator
        chunks = (
            chunk_qs.annotate(distance=CosineDistance("embedding", query_embedding))
            .order_by("distance")
            .values("id", "text", "source", "source_id", "distance")[: self.config.max_results]
        )

        for chunk in chunks:
            # Convert distance to similarity score (1 - distance for cosine)
            score = 1 - chunk["distance"]

            if score >= self.config.min_score:
                results.append(
                    MemorySearchResult(
                        content=chunk["text"],
                        score=score,
                        source=chunk["source"],
                        message_id=chunk["source_id"] if chunk["source"] == "message" else None,
                        summary_id=chunk["source_id"] if chunk["source"] == "summary" else None,
                        metadata={},
                    )
                )

        return results

    def hybrid_search(
        self,
        query: str,
        session: Session | None = None,
        agent_id: int | None = None,
        customer_id: UUID | None = None,
    ) -> list[MemorySearchResult]:
        """Perform hybrid vector + text search.

        Combines vector similarity and text matching with configurable weights.

        Args:
            query: Search query.
            session: Optional session to limit search.
            agent_id: Optional agent ID to limit search.
            customer_id: Optional customer ID to scope results to a tenant.

        Returns:
            List of search results with combined scores.
        """
        if not self.config.enabled:
            return []

        vector_weight = self.config.hybrid_weights.get("vector", 0.7)
        text_weight = self.config.hybrid_weights.get("text", 0.3)

        vector_results = self.vector_search(query, session, agent_id, customer_id)
        text_results = self.text_search(query, session, agent_id, customer_id)

        # Combine scores
        combined: dict[str, MemorySearchResult] = {}

        for result in vector_results:
            key = f"{result.source}:{result.message_id or result.summary_id}"
            result.score = result.score * vector_weight
            combined[key] = result

        for result in text_results:
            key = f"{result.source}:{result.message_id or result.summary_id}"
            if key in combined:
                combined[key].score += result.score * text_weight
            else:
                result.score = result.score * text_weight
                combined[key] = result

        results = list(combined.values())
        results.sort(key=lambda x: x.score, reverse=True)
        return results[: self.config.max_results]

    def search(
        self,
        query: str,
        session: Session | None = None,
        agent_id: int | None = None,
        customer_id: UUID | None = None,
        search_type: str = "hybrid",
    ) -> list[MemorySearchResult]:
        """Search agent memory.

        Args:
            query: Search query.
            session: Optional session to limit search.
            agent_id: Optional agent ID to limit search.
            customer_id: Optional customer ID to scope results to a tenant.
            search_type: Type of search ("hybrid", "vector", "text").

        Returns:
            List of search results.
        """
        if search_type == "vector":
            return self.vector_search(query, session, agent_id, customer_id)
        if search_type == "text":
            return self.text_search(query, session, agent_id, customer_id)
        return self.hybrid_search(query, session, agent_id, customer_id)


# Global search service instance
memory_search_service = MemorySearchService()
