"""Tests for ContextBundleService.push_conversation_summary."""

import datetime
import json
from unittest.mock import MagicMock, patch

from marvin.context import ContextBundleService
from marvin.memory.models import ConversationSummary, EmbeddingChunk


def _make_summary(session_id: str = "session-1", summary_id: str = "sum-1") -> ConversationSummary:
    chunk_id = "chunk-1"
    return ConversationSummary(
        summary_id=summary_id,
        session_id=session_id,
        summary_text="A brief summary.",
        message_count=5,
        messages=[],
        embedding_chunk_id=chunk_id,
        start_index=0,
        end_index=5,
        created_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
    )


def _make_chunk(session_id: str = "session-1", chunk_id: str = "chunk-1") -> EmbeddingChunk:
    return EmbeddingChunk(
        chunk_id=chunk_id,
        session_id=session_id,
        text="A brief summary.",
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
    )


class TestPushConversationSummary:
    def _make_s3_mock(self) -> MagicMock:
        return MagicMock()

    def test_writes_summary_and_chunk_to_s3(self) -> None:
        summary = _make_summary()
        chunk = _make_chunk()
        s3 = self._make_s3_mock()

        with patch("marvin.context.get_s3_client", return_value=s3):
            ContextBundleService().push_conversation_summary(
                s3_prefix="s3://my-bucket/customers/acme/projects/backend/",
                summary=summary,
                chunk=chunk,
            )

        assert s3.put_object.call_count == 2  # noqa: PLR2004

    def test_summary_key_path(self) -> None:
        summary = _make_summary(session_id="sess-abc", summary_id="sum-xyz")
        chunk = _make_chunk(session_id="sess-abc", chunk_id="chunk-xyz")
        s3 = self._make_s3_mock()

        with patch("marvin.context.get_s3_client", return_value=s3):
            ContextBundleService().push_conversation_summary(
                s3_prefix="s3://bucket/prefix/",
                summary=summary,
                chunk=chunk,
            )

        calls = {call[1]["Key"]: call[1] for call in s3.put_object.call_args_list}
        expected_summary_key = "prefix/summaries/sess-abc/sum-xyz.json"
        expected_chunk_key = "prefix/summaries/sess-abc/chunks/chunk-xyz.json"
        assert expected_summary_key in calls
        assert expected_chunk_key in calls

    def test_summary_body_is_valid_json(self) -> None:
        summary = _make_summary()
        chunk = _make_chunk()
        s3 = self._make_s3_mock()

        with patch("marvin.context.get_s3_client", return_value=s3):
            ContextBundleService().push_conversation_summary(
                s3_prefix="s3://bucket/prefix/",
                summary=summary,
                chunk=chunk,
            )

        calls = {call[1]["Key"]: call[1] for call in s3.put_object.call_args_list}
        summary_body = calls["prefix/summaries/session-1/sum-1.json"]["Body"]
        data = json.loads(summary_body.decode("utf-8"))
        assert data["summary_id"] == "sum-1"
        assert data["session_id"] == "session-1"
        assert data["summary_text"] == "A brief summary."

    def test_chunk_body_is_valid_json(self) -> None:
        summary = _make_summary()
        chunk = _make_chunk()
        s3 = self._make_s3_mock()

        with patch("marvin.context.get_s3_client", return_value=s3):
            ContextBundleService().push_conversation_summary(
                s3_prefix="s3://bucket/prefix/",
                summary=summary,
                chunk=chunk,
            )

        calls = {call[1]["Key"]: call[1] for call in s3.put_object.call_args_list}
        chunk_body = calls["prefix/summaries/session-1/chunks/chunk-1.json"]["Body"]
        data = json.loads(chunk_body.decode("utf-8"))
        assert data["chunk_id"] == "chunk-1"
        assert data["embedding"] == [0.1, 0.2, 0.3]

    def test_content_type_is_json(self) -> None:
        summary = _make_summary()
        chunk = _make_chunk()
        s3 = self._make_s3_mock()

        with patch("marvin.context.get_s3_client", return_value=s3):
            ContextBundleService().push_conversation_summary(
                s3_prefix="s3://bucket/prefix/",
                summary=summary,
                chunk=chunk,
            )

        for call in s3.put_object.call_args_list:
            assert call[1]["ContentType"] == "application/json"
