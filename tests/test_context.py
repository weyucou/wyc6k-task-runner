"""Tests for ContextBundleService conversation summary methods."""

import datetime
import json
from unittest.mock import MagicMock, patch

from marvin.context import ContextBundleService
from marvin.memory.models import ConversationSummary


def _make_summary(session_id: str = "session-1", summary_id: str = "sum-1") -> ConversationSummary:
    return ConversationSummary(
        summary_id=summary_id,
        session_id=session_id,
        summary_text="A brief summary.",
        message_count=5,
        messages=[],
        start_index=0,
        end_index=4,
        created_at=datetime.datetime(2026, 1, 1),
    )


class TestPushConversationSummary:
    def test_writes_summary_to_s3(self) -> None:
        summary = _make_summary()
        s3 = MagicMock()

        with patch("marvin.context.get_s3_client", return_value=s3):
            ContextBundleService().push_conversation_summary(
                s3_prefix="s3://my-bucket/customers/acme/projects/backend/",
                summary=summary,
            )

        assert s3.put_object.call_count == 1

    def test_summary_key_path(self) -> None:
        summary = _make_summary(session_id="sess-abc", summary_id="sum-xyz")
        s3 = MagicMock()

        with patch("marvin.context.get_s3_client", return_value=s3):
            ContextBundleService().push_conversation_summary(
                s3_prefix="s3://bucket/prefix/",
                summary=summary,
            )

        call_kwargs = s3.put_object.call_args[1]
        assert call_kwargs["Key"] == "prefix/conversations/sess-abc/summaries/sum-xyz.json"

    def test_summary_body_is_valid_json(self) -> None:
        summary = _make_summary()
        s3 = MagicMock()

        with patch("marvin.context.get_s3_client", return_value=s3):
            ContextBundleService().push_conversation_summary(
                s3_prefix="s3://bucket/prefix/",
                summary=summary,
            )

        call_kwargs = s3.put_object.call_args[1]
        data = json.loads(call_kwargs["Body"].decode("utf-8"))
        assert data["summary_id"] == "sum-1"
        assert data["session_id"] == "session-1"
        assert data["summary_text"] == "A brief summary."

    def test_content_type_is_json(self) -> None:
        summary = _make_summary()
        s3 = MagicMock()

        with patch("marvin.context.get_s3_client", return_value=s3):
            ContextBundleService().push_conversation_summary(
                s3_prefix="s3://bucket/prefix/",
                summary=summary,
            )

        call_kwargs = s3.put_object.call_args[1]
        assert call_kwargs["ContentType"] == "application/json"


class TestPullConversationSummaries:
    def test_returns_empty_list_when_no_objects(self) -> None:
        s3 = MagicMock()
        paginator = MagicMock()
        s3.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{"Contents": []}]

        with patch("marvin.context.get_s3_client", return_value=s3):
            result = ContextBundleService().pull_conversation_summaries(
                s3_prefix="s3://bucket/prefix/",
                session_id="session-1",
            )

        assert result == []

    def test_returns_summaries_from_s3(self) -> None:
        summary = _make_summary()
        summary_json = summary.model_dump_json().encode("utf-8")

        s3 = MagicMock()
        paginator = MagicMock()
        s3.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "prefix/conversations/session-1/summaries/sum-1.json"}]}
        ]
        s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=summary_json))}

        with patch("marvin.context.get_s3_client", return_value=s3):
            result = ContextBundleService().pull_conversation_summaries(
                s3_prefix="s3://bucket/prefix/",
                session_id="session-1",
            )

        assert len(result) == 1
        assert result[0].summary_id == "sum-1"
        assert result[0].session_id == "session-1"

    def test_uses_correct_s3_prefix(self) -> None:
        s3 = MagicMock()
        paginator = MagicMock()
        s3.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{}]

        with patch("marvin.context.get_s3_client", return_value=s3):
            ContextBundleService().pull_conversation_summaries(
                s3_prefix="s3://bucket/myprefix/",
                session_id="sess-42",
            )

        call_kwargs = paginator.paginate.call_args[1]
        assert call_kwargs["Prefix"] == "myprefix/conversations/sess-42/summaries/"
