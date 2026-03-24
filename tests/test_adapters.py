"""Tests for marvin.adapters — AdapterFactory, LambdaAdapter, FargateAdapter, BatchAdapter."""

import asyncio
from unittest.mock import MagicMock, patch

from marvin.adapters.batch_adapter import BatchAdapter
from marvin.adapters.factory import AdapterFactory
from marvin.adapters.fargate_adapter import FargateAdapter
from marvin.adapters.lambda_adapter import LambdaAdapter
from marvin.models import AgentConfig, TaskEnvelope


def _make_envelope(duration: int = 0) -> TaskEnvelope:
    return TaskEnvelope(
        task_id="t-001",
        customer_id="c-001",
        session_id="s-001",
        agent=AgentConfig(name="test-agent"),
        s3_context_prefix="s3://bucket/prefix",
        user_message="Hello",
        duration_hint_seconds=duration,
    )


class TestAdapterFactory:
    def test_select_lambda_for_short_tasks(self) -> None:
        adapter = AdapterFactory.select(_make_envelope(100))
        assert isinstance(adapter, LambdaAdapter)

    def test_select_lambda_at_upper_boundary(self) -> None:
        adapter = AdapterFactory.select(_make_envelope(599))
        assert isinstance(adapter, LambdaAdapter)

    def test_select_fargate_at_lower_boundary(self) -> None:
        adapter = AdapterFactory.select(_make_envelope(600))
        assert isinstance(adapter, FargateAdapter)

    def test_select_fargate_for_medium_tasks(self) -> None:
        adapter = AdapterFactory.select(_make_envelope(3600))
        assert isinstance(adapter, FargateAdapter)

    def test_select_fargate_at_upper_boundary(self) -> None:
        adapter = AdapterFactory.select(_make_envelope(21600))
        assert isinstance(adapter, FargateAdapter)

    def test_select_batch_above_fargate_threshold(self) -> None:
        adapter = AdapterFactory.select(_make_envelope(21601))
        assert isinstance(adapter, BatchAdapter)

    def test_select_batch_for_long_tasks(self) -> None:
        adapter = AdapterFactory.select(_make_envelope(86400))
        assert isinstance(adapter, BatchAdapter)


class TestLambdaAdapter:
    def test_dispatch_returns_inline_complete(self) -> None:
        adapter = LambdaAdapter()
        result = asyncio.run(adapter.dispatch(_make_envelope(100)))
        assert result.adapter == "lambda"
        assert result.status == "inline_complete"
        assert result.job_id is None


class TestFargateAdapter:
    def test_dispatch_calls_run_task_and_returns_job_id(self) -> None:
        mock_ecs = MagicMock()
        mock_ecs.run_task.return_value = {
            "tasks": [{"taskArn": "arn:aws:ecs:us-east-1:123456789012:task/my-cluster/abc123def456"}]
        }
        with patch("marvin.adapters.fargate_adapter.boto3.client", return_value=mock_ecs):
            adapter = FargateAdapter(
                task_definition_arn="arn:aws:ecs:us-east-1:123:task-definition/worker:1",
                cluster="my-cluster",
            )
            result = asyncio.run(adapter.dispatch(_make_envelope(1000)))

        mock_ecs.run_task.assert_called_once()
        call_kwargs = mock_ecs.run_task.call_args[1]
        assert call_kwargs["taskDefinition"] == "arn:aws:ecs:us-east-1:123:task-definition/worker:1"
        assert call_kwargs["cluster"] == "my-cluster"
        assert call_kwargs["launchType"] == "FARGATE"
        assert result.adapter == "fargate"
        assert result.status == "dispatched"
        assert result.job_id == "abc123def456"

    def test_dispatch_returns_failed_when_no_tasks(self) -> None:
        mock_ecs = MagicMock()
        mock_ecs.run_task.return_value = {"tasks": []}
        with patch("marvin.adapters.fargate_adapter.boto3.client", return_value=mock_ecs):
            adapter = FargateAdapter(task_definition_arn="arn:task-def", cluster="cluster")
            result = asyncio.run(adapter.dispatch(_make_envelope(1000)))

        assert result.status == "failed"
        assert result.job_id is None

    def test_dispatch_passes_task_envelope_as_env_var(self) -> None:
        mock_ecs = MagicMock()
        mock_ecs.run_task.return_value = {"tasks": [{"taskArn": "arn:aws:ecs:us-east-1:123:task/cluster/jobid"}]}
        envelope = _make_envelope(1000)
        with patch("marvin.adapters.fargate_adapter.boto3.client", return_value=mock_ecs):
            adapter = FargateAdapter(task_definition_arn="arn:task", cluster="c")
            asyncio.run(adapter.dispatch(envelope))

        call_kwargs = mock_ecs.run_task.call_args[1]
        container_overrides = call_kwargs["overrides"]["containerOverrides"]
        env_vars = {e["name"]: e["value"] for e in container_overrides[0]["environment"]}
        assert "TASK_ENVELOPE" in env_vars
        assert envelope.task_id in env_vars["TASK_ENVELOPE"]


class TestBatchAdapter:
    def test_dispatch_calls_submit_job_and_returns_job_id(self) -> None:
        mock_batch = MagicMock()
        mock_batch.submit_job.return_value = {"jobId": "batch-job-001"}
        with patch("marvin.adapters.batch_adapter.boto3.client", return_value=mock_batch):
            adapter = BatchAdapter(job_definition="marvin-worker-job", job_queue="marvin-queue")
            result = asyncio.run(adapter.dispatch(_make_envelope(30000)))

        mock_batch.submit_job.assert_called_once()
        call_kwargs = mock_batch.submit_job.call_args[1]
        assert call_kwargs["jobDefinition"] == "marvin-worker-job"
        assert call_kwargs["jobQueue"] == "marvin-queue"
        assert result.adapter == "batch"
        assert result.status == "dispatched"
        assert result.job_id == "batch-job-001"

    def test_dispatch_returns_failed_when_no_job_id(self) -> None:
        mock_batch = MagicMock()
        mock_batch.submit_job.return_value = {}
        with patch("marvin.adapters.batch_adapter.boto3.client", return_value=mock_batch):
            adapter = BatchAdapter(job_definition="job-def", job_queue="queue")
            result = asyncio.run(adapter.dispatch(_make_envelope(30000)))

        assert result.status == "failed"
        assert result.job_id is None

    def test_dispatch_passes_task_envelope_as_env_var(self) -> None:
        mock_batch = MagicMock()
        mock_batch.submit_job.return_value = {"jobId": "batch-job-002"}
        envelope = _make_envelope(30000)
        with patch("marvin.adapters.batch_adapter.boto3.client", return_value=mock_batch):
            adapter = BatchAdapter(job_definition="job-def", job_queue="queue")
            asyncio.run(adapter.dispatch(envelope))

        call_kwargs = mock_batch.submit_job.call_args[1]
        env_vars = {e["name"]: e["value"] for e in call_kwargs["containerOverrides"]["environment"]}
        assert "TASK_ENVELOPE" in env_vars
        assert envelope.task_id in env_vars["TASK_ENVELOPE"]
