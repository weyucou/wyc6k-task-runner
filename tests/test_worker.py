# Copyright (c) 2026 MNKT
"""Tests for marvin.worker — the in-process SQS worker loop."""

import asyncio
import datetime as dt
import json
import logging
import signal
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import boto3
import pytest
from moto import mock_aws

import entrypoint
from marvin import worker
from marvin.context import CustomerContextBundle, MemoryEntry
from marvin.models import AgentConfig, TaskEnvelope
from tests.fake_llm import FakeLLMClient, fake_text

if TYPE_CHECKING:
    from collections.abc import Coroutine, Iterator

LEAK_MARKER = "sk-ant-LEAKMARKER"
# No user_message, and agent.api_key ends the JSON: pydantic's truncated input_value keeps the input's tail.
LEAKY_ENVELOPE_JSON = json.dumps(
    {
        "task_id": "task-001",
        "customer_id": "cust-001",
        "session_id": "sess-001",
        "s3_context_prefix": "s3://bucket/customers/cust-001/projects/repo",
        "agent": {"name": "test-agent", "api_key": LEAK_MARKER},
    }
)


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
        [(LEAKY_ENVELOPE_JSON, "user_message"), (LEAKY_ENVELOPE_JSON[:-1], "json_invalid")],
        ids=["missing-field", "truncated-json"],
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


class TestProcessEnvelope:
    def test_background_processes_stopped_when_the_run_raises(self) -> None:
        envelope = TaskEnvelope(
            task_id="task-001",
            customer_id="cust-001",
            session_id="sess-001",
            agent=AgentConfig(name="test-agent"),
            s3_context_prefix="s3://bucket/customers/cust-001/projects/repo",
            user_message="Hello, agent!",
        )
        with (
            patch("marvin.worker.ContextBundleService"),
            patch("marvin.worker.AgentRunner") as mock_runner_cls,
            patch("marvin.worker.stop_background_processes") as mock_stop,
        ):
            mock_runner_cls.return_value.chat = AsyncMock(side_effect=RuntimeError("LLM timeout"))
            with pytest.raises(RuntimeError, match="LLM timeout"):
                asyncio.run(worker.process_envelope(envelope))
        mock_stop.assert_called_once_with()


def _context_bundle() -> CustomerContextBundle:
    return CustomerContextBundle(
        customer_id="cust-001",
        claude_md="Customer rules.",
        sops={"deploy.md": "Deploy SOP."},
        project_goals="Project goals.",
        memory_index="Memory index.",
        daily_memories=[MemoryEntry(date=dt.date(2026, 11, 12), filename="2026-11-12.md", content="Previous run.")],
    )


def _system_prompt_sent(run: Coroutine[Any, Any, Any]) -> str | None:
    """Run `run` against a scripted model and return the system prompt of its one request."""
    client = FakeLLMClient([fake_text("done")])
    with patch("marvin.runner.create_client_from_agent_config", return_value=client):
        asyncio.run(run)
    [request] = client.requests
    return request.system_prompt


class TestProcessEnvelopeSystemPrompt:
    def _envelope(self) -> TaskEnvelope:
        return TaskEnvelope(
            task_id="task-001",
            customer_id="cust-001",
            session_id="sess-001",
            agent=AgentConfig(name="test-agent", system_prompt="Agent prompt."),
            s3_context_prefix="s3://bucket/customers/cust-001/projects/repo",
            user_message="Hello, agent!",
        )

    def test_includes_the_context_bundle_and_the_agent_prompt(self) -> None:
        with patch("marvin.worker.ContextBundleService") as service_cls:
            service_cls.return_value.pull.return_value = _context_bundle()
            prompt = _system_prompt_sent(worker.process_envelope(self._envelope()))
        assert prompt is not None
        assert "Agent prompt." in prompt
        for text in ("Customer rules.", "Deploy SOP.", "Project goals.", "Memory index.", "Previous run."):
            assert text in prompt
            assert prompt.index(text) < prompt.index("Agent prompt.")

    def test_matches_the_entrypoint_system_prompt(self) -> None:
        envelope, bundle = self._envelope(), _context_bundle()
        with patch("marvin.worker.ContextBundleService") as service_cls:
            service_cls.return_value.pull.return_value = bundle
            worker_prompt = _system_prompt_sent(worker.process_envelope(envelope))
        assert worker_prompt == _system_prompt_sent(entrypoint._run_agent(envelope, bundle))


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
