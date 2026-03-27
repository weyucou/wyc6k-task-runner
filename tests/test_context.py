"""Tests for ContextBundleService.push_conversation_summary and pull_conversation_summaries."""

import datetime
import json
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from marvin.context import ContextBundleService
from marvin.memory.models import ConversationSummary, Message

_EXPECTED_MESSAGE_COUNT = 2


def _make_summary(session_id: str = "sess-1") -> ConversationSummary:
    return ConversationSummary(
        session_id=session_id,
        summary_text="Summary text",
        message_count=_EXPECTED_MESSAGE_COUNT,
        messages=[
            Message(role="user", content="Hello", timestamp=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)),
            Message(role="assistant", content="Hi", timestamp=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)),
        ],
        start_index=0,
        end_index=1,
    )


class TestPushConversationSummary:
    @patch("marvin.context.get_s3_client")
    def test_writes_to_correct_s3_key(self, mock_get_s3_client: MagicMock) -> None:
        mock_s3 = MagicMock()
        mock_get_s3_client.return_value = mock_s3

        service = ContextBundleService()
        summary = _make_summary()
        service.push_conversation_summary("s3://my-bucket/customers/c1/projects/proj/", summary)

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "my-bucket"
        assert f"conversations/{summary.session_id}/summaries/{summary.summary_id}.json" in call_kwargs["Key"]

    @patch("marvin.context.get_s3_client")
    def test_writes_valid_json(self, mock_get_s3_client: MagicMock) -> None:
        mock_s3 = MagicMock()
        mock_get_s3_client.return_value = mock_s3

        service = ContextBundleService()
        summary = _make_summary()
        service.push_conversation_summary("s3://my-bucket/customers/c1/projects/proj/", summary)

        call_kwargs = mock_s3.put_object.call_args[1]
        body = call_kwargs["Body"]
        parsed = json.loads(body.decode("utf-8"))
        assert parsed["session_id"] == summary.session_id
        assert parsed["summary_text"] == "Summary text"
        assert parsed["message_count"] == _EXPECTED_MESSAGE_COUNT


    @patch("marvin.context.get_s3_client")
    def test_swallows_s3_errors(self, mock_get_s3_client: MagicMock) -> None:
        mock_s3 = MagicMock()
        mock_get_s3_client.return_value = mock_s3
        mock_s3.put_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "Bucket not found"}}, "PutObject"
        )

        service = ContextBundleService()
        summary = _make_summary()
        # Must not raise — failures are logged and swallowed
        service.push_conversation_summary("s3://my-bucket/customers/c1/projects/proj/", summary)


class TestPullConversationSummaries:
    @patch("marvin.context.get_s3_client")
    def test_returns_empty_when_none_exist(self, mock_get_s3_client: MagicMock) -> None:
        mock_s3 = MagicMock()
        mock_get_s3_client.return_value = mock_s3
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{"Contents": []}]

        service = ContextBundleService()
        result = service.pull_conversation_summaries("s3://my-bucket/customers/c1/projects/proj/", "sess-1")

        assert result == []

    @patch("marvin.context.get_s3_client")
    def test_returns_summaries(self, mock_get_s3_client: MagicMock) -> None:
        summary = _make_summary()
        summary_json = summary.model_dump_json()

        mock_s3 = MagicMock()
        mock_get_s3_client.return_value = mock_s3
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "customers/c1/projects/proj/conversations/sess-1/summaries/abc.json"}]}
        ]
        mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: summary_json.encode("utf-8"))}

        service = ContextBundleService()
        result = service.pull_conversation_summaries("s3://my-bucket/customers/c1/projects/proj/", "sess-1")

        assert len(result) == 1
        assert result[0].session_id == summary.session_id
        assert result[0].summary_text == "Summary text"

    @patch("marvin.context.get_s3_client")
    def test_skips_empty_content(self, mock_get_s3_client: MagicMock) -> None:
        mock_s3 = MagicMock()
        mock_get_s3_client.return_value = mock_s3
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "customers/c1/projects/proj/conversations/sess-1/summaries/empty.json"}]}
        ]
        # _read_object returns "" for NoSuchKey
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject"
        )

        service = ContextBundleService()
        result = service.pull_conversation_summaries("s3://my-bucket/customers/c1/projects/proj/", "sess-1")

        assert result == []
