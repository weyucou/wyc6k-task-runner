# Copyright (c) 2026 MNKT
"""Tests for marvin.worker — the in-process SQS worker loop."""

import asyncio
import json
import logging
import signal
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import boto3
import pytest
from moto import mock_aws

from marvin import worker
from marvin.models import TaskEnvelope

if TYPE_CHECKING:
    from collections.abc import Iterator

LEAK_MARKER = "sk-ant-LEAKMARKER"
# agent.api_key is rejected at parse. It ends the JSON, where pydantic's truncated input_value would show it.
LEAKY_ENVELOPE_JSON = json.dumps(
    {
        "schema_version": 1,
        "task_id": "task-001",
        "customer_id": "cust-001",
        "task_source_id": 12,
        "repository": "acme-corp/billing-api",
        "issue_url": "https://github.com/acme-corp/billing-api/issues/42",
        "action": "develop",
        "context_s3_prefix": "s3://bucket/customers/cust-001/projects/billing-api/",
        "agent": {
            "provider": "anthropic",
            "model_name": "claude-sonnet-5-5",
            "tool_profile": "coding",
            "api_key": LEAK_MARKER,
        },
    }
)


def _make_envelope_dict(**kwargs: Any) -> dict[str, Any]:
    envelope = {
        "schema_version": 1,
        "task_id": "task-001",
        "customer_id": "cust-001",
        "task_source_id": 12,
        "repository": "acme-corp/billing-api",
        "issue_url": "https://github.com/acme-corp/billing-api/issues/42",
        "action": "develop",
        "agent": {"provider": "anthropic", "model_name": "claude-sonnet-5-5", "tool_profile": "coding"},
        "context_s3_prefix": "s3://bucket/customers/cust-001/projects/billing-api/",
    }
    envelope.update(kwargs)
    return envelope


@pytest.fixture(autouse=True)
def _restore_shutdown_state() -> Iterator[None]:
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    yield
    for signum, handler in handlers.items():
        signal.signal(signum, handler)
    worker._shutdown.clear()


@pytest.fixture
def sqs_queue(monkeypatch) -> Iterator[tuple[Any, str]]:
    """A moto SQS queue that poll_once reads (dummy credentials, so nothing reaches real AWS)."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        sqs = boto3.client("sqs", region_name="us-east-1")
        queue_url = sqs.create_queue(QueueName="tasks")["QueueUrl"]
        monkeypatch.setattr(worker, "SQS_QUEUE_URL", queue_url)
        yield sqs, queue_url


class TestPollOnce:
    @pytest.mark.parametrize(
        ("body", "named"),
        [(LEAKY_ENVELOPE_JSON, "agent.api_key"), (LEAKY_ENVELOPE_JSON[:-1], "json_invalid")],
        ids=["api-key-rejected", "truncated-json"],
    )
    def test_invalid_envelope_logged_without_input_and_retained(self, sqs_queue, caplog, body, named) -> None:
        sqs, queue_url = sqs_queue
        sqs.send_message(QueueUrl=queue_url, MessageBody=body)
        # marvin's loggers only: botocore's DEBUG output echoes the message body it received.
        with caplog.at_level(logging.DEBUG, logger="marvin"):
            worker.poll_once(sqs)
        assert LEAK_MARKER not in caplog.text  # caplog.text includes formatted tracebacks
        assert named in caplog.text
        attributes = sqs.get_queue_attributes(
            QueueUrl=queue_url, AttributeNames=["ApproximateNumberOfMessagesNotVisible"]
        )["Attributes"]
        assert attributes["ApproximateNumberOfMessagesNotVisible"] == "1"  # not deleted: redelivered, then DLQ

    def test_resolved_key_passed_to_process_envelope_and_message_deleted(self, sqs_queue, caplog) -> None:
        sqs, queue_url = sqs_queue
        sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(_make_envelope_dict()))
        result = {"task_id": "task-001", "response": "reply"}
        with (
            patch("marvin.worker.CredentialResolver.resolve", return_value="sk-resolved"),
            patch("marvin.worker.process_envelope", new=AsyncMock(return_value=result)) as mock_process,
            caplog.at_level(logging.INFO, logger="marvin.worker"),
        ):
            worker.poll_once(sqs)
        envelope, api_key = mock_process.call_args.args
        assert envelope == TaskEnvelope.model_validate(_make_envelope_dict())
        assert api_key == "sk-resolved"
        assert "acme-corp/billing-api" in caplog.text
        assert "https://github.com/acme-corp/billing-api/issues/42" in caplog.text
        attributes = sqs.get_queue_attributes(
            QueueUrl=queue_url, AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"]
        )["Attributes"]
        assert attributes == {"ApproximateNumberOfMessages": "0", "ApproximateNumberOfMessagesNotVisible": "0"}


class TestProcessEnvelope:
    def test_runner_gets_agent_config_and_task_prompt(self) -> None:
        envelope = TaskEnvelope.model_validate(_make_envelope_dict())
        with (
            patch("marvin.worker.ContextBundleService") as mock_svc_cls,
            patch("marvin.worker.AgentRunner") as mock_runner_cls,
        ):
            mock_runner_cls.return_value.chat = AsyncMock(return_value=("reply", []))
            asyncio.run(worker.process_envelope(envelope, "sk-resolved"))
        mock_svc_cls.return_value.pull.assert_called_once_with("s3://bucket/customers/cust-001/projects/billing-api/")
        assert mock_runner_cls.call_args.kwargs["agent"] == envelope.agent_config("sk-resolved")
        assert mock_runner_cls.return_value.chat.call_args.args[0] == envelope.task_prompt()

    def test_background_processes_stopped_when_the_run_raises(self) -> None:
        envelope = TaskEnvelope.model_validate(_make_envelope_dict())
        with (
            patch("marvin.worker.ContextBundleService"),
            patch("marvin.worker.AgentRunner") as mock_runner_cls,
            patch("marvin.worker.stop_background_processes") as mock_stop,
        ):
            mock_runner_cls.return_value.chat = AsyncMock(side_effect=RuntimeError("LLM timeout"))
            with pytest.raises(RuntimeError, match="LLM timeout"):
                asyncio.run(worker.process_envelope(envelope, None))
        mock_stop.assert_called_once_with()


class TestMain:
    @pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
    def test_signal_stops_after_current_poll(self, signum) -> None:
        polls: list[object] = []

        def poll(sqs: object) -> None:
            polls.append(sqs)
            signal.raise_signal(signum)

        with (
            patch("marvin.worker._get_sqs_client", return_value="sqs-client"),
            patch("marvin.worker.poll_once", side_effect=poll),
        ):
            worker.main()
        assert polls == ["sqs-client"]

    def test_poll_error_is_logged_and_polling_continues(self, caplog) -> None:
        polls: list[object] = []

        def poll(sqs: object) -> None:
            polls.append(sqs)
            if len(polls) == 1:
                raise RuntimeError("receive failed")
            signal.raise_signal(signal.SIGTERM)

        with (
            patch("marvin.worker._get_sqs_client", return_value="sqs-client"),
            patch("marvin.worker.poll_once", side_effect=poll),
            caplog.at_level(logging.ERROR, logger="marvin.worker"),
        ):
            worker.main()
        assert len(polls) == 2
        [record] = caplog.records
        assert record.getMessage() == "Poll error"
        assert record.exc_info is not None
        assert str(record.exc_info[1]) == "receive failed"
