"""Tests for ConversationSummarizer."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from marvin.llm.base import LLMMessage, LLMResponse, StopReason
from marvin.memory.summarizer import ConversationSummarizer
from marvin.models import AgentConfig, LLMProvider


def _make_agent() -> AgentConfig:
    return AgentConfig(
        name="test-agent",
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-sonnet-4-20250514",
    )


def _make_messages(count: int) -> list[LLMMessage]:
    messages = []
    for i in range(count):
        if i % 2 == 0:
            messages.append(LLMMessage.user(f"User message {i}"))
        else:
            messages.append(LLMMessage.assistant(f"Assistant message {i}"))
    return messages


class TestConversationSummarizer:
    def test_returns_none_when_below_threshold(self) -> None:
        summarizer = ConversationSummarizer(_make_agent(), threshold=50, batch_size=20)
        result = asyncio.run(summarizer.maybe_summarize("session-1", _make_messages(30)))
        assert result is None

    def test_returns_none_at_exact_threshold(self) -> None:
        summarizer = ConversationSummarizer(_make_agent(), threshold=50, batch_size=20)
        result = asyncio.run(summarizer.maybe_summarize("session-1", _make_messages(50)))
        assert result is None

    @patch("marvin.memory.summarizer._compute_embedding")
    @patch("marvin.memory.summarizer.create_client_from_agent_config")
    def test_creates_summary_when_above_threshold(
        self,
        mock_create_client: MagicMock,
        mock_embedding: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(
            return_value=LLMResponse(content="Summary text.", stop_reason=StopReason.END_TURN)
        )
        mock_create_client.return_value = mock_client
        mock_embedding.return_value = [0.1, 0.2, 0.3]

        summarizer = ConversationSummarizer(_make_agent(), threshold=50, batch_size=20)
        result = asyncio.run(summarizer.maybe_summarize("session-1", _make_messages(55)))

        batch_size = 20
        assert result is not None
        summary, chunk = result
        assert summary.session_id == "session-1"
        assert summary.summary_text == "Summary text."
        assert summary.message_count == batch_size
        assert summary.start_index == 0
        assert summary.end_index == batch_size
        assert summary.embedding_chunk_id == chunk.chunk_id
        assert chunk.session_id == "session-1"
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.text == "Summary text."

    @patch("marvin.memory.summarizer._compute_embedding")
    @patch("marvin.memory.summarizer.create_client_from_agent_config")
    def test_summary_preserves_original_messages(
        self,
        mock_create_client: MagicMock,
        mock_embedding: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(
            return_value=LLMResponse(content="A summary.", stop_reason=StopReason.END_TURN)
        )
        mock_create_client.return_value = mock_client
        mock_embedding.return_value = [0.5]

        summarizer = ConversationSummarizer(_make_agent(), threshold=5, batch_size=3)
        result = asyncio.run(summarizer.maybe_summarize("session-2", _make_messages(10)))

        batch_size = 3
        assert result is not None
        summary, _ = result
        assert len(summary.messages) == batch_size
        assert summary.messages[0].role == "user"
        assert summary.messages[0].content == "User message 0"
        assert summary.messages[1].role == "assistant"
        assert summary.messages[1].content == "Assistant message 1"

    @patch("marvin.memory.summarizer._compute_embedding")
    @patch("marvin.memory.summarizer.create_client_from_agent_config")
    def test_summary_and_chunk_share_chunk_id(
        self,
        mock_create_client: MagicMock,
        mock_embedding: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=LLMResponse(content="Linked.", stop_reason=StopReason.END_TURN))
        mock_create_client.return_value = mock_client
        mock_embedding.return_value = [0.0]

        summarizer = ConversationSummarizer(_make_agent(), threshold=2, batch_size=2)
        result = asyncio.run(summarizer.maybe_summarize("s", _make_messages(5)))

        assert result is not None
        summary, chunk = result
        assert summary.embedding_chunk_id == chunk.chunk_id

    @patch("marvin.memory.summarizer._compute_embedding")
    @patch("marvin.memory.summarizer.create_client_from_agent_config")
    def test_uses_configured_batch_size(
        self,
        mock_create_client: MagicMock,
        mock_embedding: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(
            return_value=LLMResponse(content="Batch summary.", stop_reason=StopReason.END_TURN)
        )
        mock_create_client.return_value = mock_client
        mock_embedding.return_value = []

        summarizer = ConversationSummarizer(_make_agent(), threshold=10, batch_size=5)
        result = asyncio.run(summarizer.maybe_summarize("s", _make_messages(15)))

        batch_size = 5
        assert result is not None
        summary, _ = result
        assert summary.message_count == batch_size
        assert len(summary.messages) == batch_size
