"""Unit tests for ConversationSummarizer."""

import asyncio
import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from marvin.llm.base import LLMMessage, LLMResponse, MessageRole, StopReason
from marvin.memory.models import ConversationSummary, Message
from marvin.memory.summarizer import ConversationSummarizer


def _make_messages(n: int) -> list[LLMMessage]:
    messages = []
    for i in range(n):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        if role == MessageRole.USER:
            messages.append(LLMMessage.user(f"Message {i}"))
        else:
            messages.append(LLMMessage.assistant(f"Message {i}"))
    return messages


def _make_mock_client(summary_text: str = "Summary of conversation.") -> MagicMock:
    client = MagicMock()
    response = LLMResponse(content=summary_text, stop_reason=StopReason.END_TURN)
    client.generate = AsyncMock(return_value=response)
    return client


class TestConversationSummarizerBelowThreshold:
    def test_returns_none_when_below_threshold(self) -> None:
        client = _make_mock_client()
        summarizer = ConversationSummarizer(client=client, threshold=10, batch_size=5)
        messages = _make_messages(5)

        result = asyncio.run(summarizer.maybe_summarize("session-1", messages))

        assert result is None
        client.generate.assert_not_called()

    def test_returns_none_when_equal_to_threshold(self) -> None:
        client = _make_mock_client()
        summarizer = ConversationSummarizer(client=client, threshold=10, batch_size=5)
        messages = _make_messages(10)

        result = asyncio.run(summarizer.maybe_summarize("session-1", messages))

        assert result is None


class TestConversationSummarizerAboveThreshold:
    def test_returns_summary_when_above_threshold(self) -> None:
        client = _make_mock_client("This is the summary.")
        summarizer = ConversationSummarizer(client=client, threshold=5, batch_size=3)
        messages = _make_messages(10)

        result = asyncio.run(summarizer.maybe_summarize("session-abc", messages))

        assert result is not None
        assert isinstance(result, ConversationSummary)
        assert result.session_id == "session-abc"
        assert result.summary_text == "This is the summary."

    def test_summary_contains_original_messages(self) -> None:
        client = _make_mock_client()
        summarizer = ConversationSummarizer(client=client, threshold=5, batch_size=3)
        messages = _make_messages(10)

        result = asyncio.run(summarizer.maybe_summarize("session-1", messages))

        assert result is not None
        assert result.message_count == 3
        assert len(result.messages) == 3
        assert all(isinstance(m, Message) for m in result.messages)

    def test_summary_indices_cover_batch(self) -> None:
        client = _make_mock_client()
        summarizer = ConversationSummarizer(client=client, threshold=5, batch_size=4)
        messages = _make_messages(10)

        result = asyncio.run(summarizer.maybe_summarize("session-1", messages))

        assert result is not None
        assert result.start_index == 0
        assert result.end_index == 3

    def test_llm_called_with_low_temperature(self) -> None:
        client = _make_mock_client()
        summarizer = ConversationSummarizer(client=client, threshold=5, batch_size=3)
        messages = _make_messages(10)

        asyncio.run(summarizer.maybe_summarize("session-1", messages))

        call_kwargs = client.generate.call_args[1]
        assert call_kwargs.get("temperature") == 0.3

    def test_does_not_modify_original_messages(self) -> None:
        client = _make_mock_client()
        summarizer = ConversationSummarizer(client=client, threshold=5, batch_size=3)
        messages = _make_messages(10)
        original_len = len(messages)

        asyncio.run(summarizer.maybe_summarize("session-1", messages))

        assert len(messages) == original_len


class TestConversationSummaryModel:
    def test_summary_id_auto_generated(self) -> None:
        summary = ConversationSummary(
            session_id="s1",
            summary_text="text",
            message_count=1,
            messages=[Message(role="user", content="hi", timestamp=datetime.datetime.now(datetime.UTC))],
            start_index=0,
            end_index=0,
        )
        assert summary.summary_id
        assert len(summary.summary_id) > 0

    def test_two_summaries_have_different_ids(self) -> None:
        kwargs: Any = {
            "session_id": "s1",
            "summary_text": "text",
            "message_count": 1,
            "messages": [Message(role="user", content="hi", timestamp=datetime.datetime.now(datetime.UTC))],
            "start_index": 0,
            "end_index": 0,
        }
        s1 = ConversationSummary(**kwargs)
        s2 = ConversationSummary(**kwargs)
        assert s1.summary_id != s2.summary_id
