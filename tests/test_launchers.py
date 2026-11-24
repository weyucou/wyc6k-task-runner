# Copyright (c) 2026 MNKT
"""Tests for marvin.launchers — backend selection, docker/ecs launch, SQS consumer."""

import json
import signal
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import boto3
import pytest
import structlog
from botocore.exceptions import ClientError
from moto import mock_aws
from structlog.testing import capture_logs

from marvin.launchers import LaunchResult, get_launcher
from marvin.launchers.consumer import extend_visibility, main, poll_once
from marvin.launchers.docker import DEFAULT_ENV_PASSTHROUGH, DockerLauncher, build_docker_command
from marvin.launchers.ecs import EcsLauncher, result_from_run_task
from marvin.models import TaskEnvelope

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

REGION = "us-east-1"
ECS_ENV = {
    "ECS_CLUSTER": "agent-worker",
    "ECS_TASK_DEFINITION": "agent-worker:1",
    "ECS_SUBNETS": "subnet-a, subnet-b",
    "ECS_SECURITY_GROUPS": "sg-a",
}
PRE_CONTRACT_ENVELOPE_JSON = (Path(__file__).parent / "fixtures" / "envelopes" / "pre_contract_sender.json").read_text(
    encoding="utf-8"
)


def _envelope_json(**kwargs: Any) -> str:
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
    return json.dumps(envelope)


def _forwarded_env_names(cmd: list[str]) -> list[str]:
    return [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-e"]


def _message_counts(sqs: Any, queue_url: str) -> tuple[int, int]:
    """Return (visible, in flight) message counts."""
    names = ["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"]
    attributes = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=names)["Attributes"]
    return int(attributes[names[0]]), int(attributes[names[1]])


class FakeLauncher:
    """Succeeds unless the envelope's task_id is in `failing`; records the raw bodies it launched."""

    def __init__(
        self,
        failing: frozenset[str] = frozenset(),
        heartbeat: bool = False,
        on_launch: Callable[[], object] = lambda: None,
    ) -> None:
        self.failing = failing
        self.heartbeat = heartbeat
        self.on_launch = on_launch  # runs while the "task" runs
        self.launched: list[str] = []

    def launch(self, envelope: TaskEnvelope, raw_json: str, heartbeat: Callable[[], None]) -> LaunchResult:
        self.launched.append(raw_json)
        self.on_launch()
        if self.heartbeat:
            heartbeat()
        failed = envelope.task_id in self.failing
        return LaunchResult(success=not failed, detail="exit=1" if failed else "exit=0")


@pytest.fixture(autouse=True)
def _isolate_structlog() -> Iterator[None]:
    """main() configures structlog globally and binds context vars; start and end each test clean."""
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
    yield
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()


@pytest.fixture
def moto_region(monkeypatch) -> Iterator[str]:
    """Mock all AWS calls with moto (dummy credentials, so nothing reaches real AWS); yield the region."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("SQS_ENDPOINT_URL", raising=False)  # moto only intercepts AWS endpoints
    with mock_aws():
        yield REGION


@pytest.fixture
def docker_on_path() -> Iterator[None]:
    with patch("marvin.launchers.docker.shutil.which", return_value="/usr/bin/docker"):
        yield


@pytest.fixture
def ecs_env(monkeypatch) -> None:
    for name, value in ECS_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("ECS_ASSIGN_PUBLIC_IP", raising=False)
    monkeypatch.delenv("ECS_CONTAINER_NAME", raising=False)


@pytest.fixture
def ecs_cluster(moto_region) -> dict[str, Any]:
    """Moto Fargate cluster + awsvpc task definition shaped like wyc6k-infra's agent-worker."""
    ec2 = boto3.client("ec2", region_name=moto_region)
    vpc_id = ec2.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]["VpcId"]
    # moto gives a Fargate task ENI the private DNS name it needs only when DNS hostnames are enabled.
    ec2.modify_vpc_attribute(VpcId=vpc_id, EnableDnsHostnames={"Value": True})
    subnet_id = ec2.create_subnet(VpcId=vpc_id, CidrBlock="10.0.1.0/24")["Subnet"]["SubnetId"]
    group_id = ec2.create_security_group(GroupName="agent-worker", Description="agent-worker", VpcId=vpc_id)["GroupId"]
    ecs = boto3.client("ecs", region_name=moto_region)
    cluster_arn = ecs.create_cluster(clusterName="agent-worker")["cluster"]["clusterArn"]
    task_definition_arn = ecs.register_task_definition(
        family="agent-worker",
        networkMode="awsvpc",
        requiresCompatibilities=["FARGATE"],
        cpu="1024",
        memory="2048",
        containerDefinitions=[{"name": "agent-worker", "image": "weyucou/agent-worker:latest", "memory": 2048}],
    )["taskDefinition"]["taskDefinitionArn"]
    launcher = EcsLauncher(
        client=ecs,
        cluster=cluster_arn,
        task_definition=task_definition_arn,
        subnets=[subnet_id],
        security_groups=[group_id],
    )
    return {
        "ecs": ecs,
        "ec2": ec2,
        "launcher": launcher,
        "cluster_arn": cluster_arn,
        "task_definition_arn": task_definition_arn,
        "subnet_id": subnet_id,
        "group_id": group_id,
    }


@pytest.fixture
def sqs_queue(moto_region) -> tuple[Any, str]:
    sqs = boto3.client("sqs", region_name=moto_region)
    return sqs, sqs.create_queue(QueueName="tasks")["QueueUrl"]


class TestGetLauncher:
    def test_defaults_to_docker(self, monkeypatch, docker_on_path) -> None:
        monkeypatch.delenv("COMPUTE_BACKEND", raising=False)
        assert isinstance(get_launcher(), DockerLauncher)

    def test_docker_backend(self, monkeypatch, docker_on_path) -> None:
        monkeypatch.setenv("COMPUTE_BACKEND", "docker")
        assert isinstance(get_launcher(), DockerLauncher)

    def test_ecs_backend(self, monkeypatch, ecs_env, moto_region) -> None:
        monkeypatch.setenv("COMPUTE_BACKEND", "ecs")
        assert isinstance(get_launcher(), EcsLauncher)

    def test_argument_overrides_env(self, monkeypatch, docker_on_path) -> None:
        monkeypatch.setenv("COMPUTE_BACKEND", "ecs")
        assert isinstance(get_launcher("docker"), DockerLauncher)

    @pytest.mark.parametrize("value", ["ECS", "lambda", ""])
    def test_invalid_backend_fails_fast(self, monkeypatch, value) -> None:
        monkeypatch.setenv("COMPUTE_BACKEND", value)
        with pytest.raises(ValueError, match=r"Invalid COMPUTE_BACKEND='.*'; expected one of: docker, ecs"):
            get_launcher()

    def test_ecs_missing_config_fails_fast(self, monkeypatch, moto_region) -> None:
        monkeypatch.setenv("COMPUTE_BACKEND", "ecs")
        for name in ECS_ENV:
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(ValueError, match="ECS_CLUSTER, ECS_TASK_DEFINITION, ECS_SUBNETS, ECS_SECURITY_GROUPS"):
            get_launcher()

    def test_docker_cli_missing_fails_fast(self, monkeypatch) -> None:
        monkeypatch.setenv("COMPUTE_BACKEND", "docker")
        with (
            patch("marvin.launchers.docker.shutil.which", return_value=None),
            pytest.raises(ValueError, match="docker CLI not found"),
        ):
            get_launcher()


class TestDockerLauncherFromEnv:
    def test_defaults(self, monkeypatch, docker_on_path) -> None:
        for name in ("AGENT_WORKER_IMAGE", "DOCKER_NETWORK", "DOCKER_AWS_CONFIG_DIR", "DOCKER_ENV_PASSTHROUGH"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv("VISIBILITY_TIMEOUT", raising=False)
        launcher = DockerLauncher.from_env()
        assert launcher.image == "weyucou/agent-worker:latest"
        assert launcher.env_passthrough == DEFAULT_ENV_PASSTHROUGH
        assert launcher.network == ""
        assert launcher.aws_config_dir == ""
        assert launcher.heartbeat_seconds == 150  # half the default 300 s VISIBILITY_TIMEOUT

    def test_reads_env(self, monkeypatch, docker_on_path) -> None:
        monkeypatch.setenv("HOME", "/home/tester")
        monkeypatch.setenv("AGENT_WORKER_IMAGE", "weyucou/agent-worker:local")
        monkeypatch.setenv("DOCKER_NETWORK", "wade_default")
        monkeypatch.setenv("DOCKER_AWS_CONFIG_DIR", "~/.aws")
        monkeypatch.setenv("DOCKER_ENV_PASSTHROUGH", "EXTRA_ONE, EXTRA_TWO,")
        monkeypatch.setenv("VISIBILITY_TIMEOUT", "120")
        launcher = DockerLauncher.from_env()
        assert launcher.image == "weyucou/agent-worker:local"
        assert launcher.network == "wade_default"
        assert launcher.aws_config_dir == "/home/tester/.aws"
        assert launcher.env_passthrough == (*DEFAULT_ENV_PASSTHROUGH, "EXTRA_ONE", "EXTRA_TWO")
        assert launcher.heartbeat_seconds == 60


class TestEcsLauncherFromEnv:
    def test_reads_env(self, ecs_env, moto_region) -> None:
        launcher = EcsLauncher.from_env()
        assert launcher.cluster == "agent-worker"
        assert launcher.task_definition == "agent-worker:1"
        assert launcher.subnets == ["subnet-a", "subnet-b"]
        assert launcher.security_groups == ["sg-a"]
        assert launcher.assign_public_ip == "DISABLED"
        assert launcher.container_name == "agent-worker"

    def test_blank_list_counts_as_missing(self, monkeypatch, ecs_env, moto_region) -> None:
        monkeypatch.setenv("ECS_SUBNETS", " , ")
        with pytest.raises(ValueError, match="requires ECS_SUBNETS$"):
            EcsLauncher.from_env()


class TestBuildDockerCommand:
    def test_minimal_command(self) -> None:
        assert build_docker_command("img:tag", ["TASK_ENVELOPE_JSON"]) == [
            "docker",
            "run",
            "--rm",
            "-e",
            "TASK_ENVELOPE_JSON",
            "img:tag",
        ]

    def test_network_and_aws_config_mount(self) -> None:
        cmd = build_docker_command("img", ["A", "B"], network="wade_default", aws_config_dir="/home/tester/.aws")
        assert cmd == [
            "docker",
            "run",
            "--rm",
            "--network",
            "wade_default",
            "-v",
            "/home/tester/.aws:/root/.aws:ro",
            "-e",
            "A",
            "-e",
            "B",
            "img",
        ]


class TestDockerLauncherLaunch:
    RAW = _envelope_json()

    def _launch(
        self,
        launcher: DockerLauncher | None = None,
        wait: tuple[int | Exception, ...] = (0,),
        heartbeat: Callable[[], None] = lambda: None,
    ) -> tuple[LaunchResult, list[str], dict[str, str]]:
        """Launch with Popen patched; return (result, argv, subprocess env)."""
        launcher = launcher or DockerLauncher(image="img")
        with patch("marvin.launchers.docker.subprocess.Popen") as mock_popen:
            mock_popen.return_value.wait.side_effect = list(wait)
            result = launcher.launch(TaskEnvelope.model_validate_json(self.RAW), self.RAW, heartbeat)
        return result, mock_popen.call_args.args[0], mock_popen.call_args.kwargs["env"]

    @pytest.mark.parametrize(("returncode", "success"), [(0, True), (1, False), (2, False), (125, False)])
    def test_exit_code_decides_success(self, returncode, success) -> None:
        result, _, _ = self._launch(wait=(returncode,))
        assert result.success is success
        assert result.detail == f"exit={returncode}"

    def test_envelope_passed_unchanged_via_env(self, monkeypatch) -> None:
        monkeypatch.setenv("TASK_ENVELOPE_JSON", "stale")
        monkeypatch.setenv("CUSTOMER_ID", "other-customer")
        _, cmd, env = self._launch()
        assert env["TASK_ENVELOPE_JSON"] == self.RAW
        assert env["CUSTOMER_ID"] == "cust-001"
        assert _forwarded_env_names(cmd)[:2] == ["TASK_ENVELOPE_JSON", "CUSTOMER_ID"]

    def test_values_never_in_argv(self, monkeypatch) -> None:
        secrets = {"AWS_SECRET_ACCESS_KEY": "aws-secret-value", "ANTHROPIC_API_KEY": "sk-ant-value"}
        for name, value in secrets.items():
            monkeypatch.setenv(name, value)
        _, cmd, env = self._launch()
        assert set(secrets) <= set(_forwarded_env_names(cmd))
        assert all("=" not in name for name in _forwarded_env_names(cmd))
        for value in (*secrets.values(), self.RAW):
            assert not any(value in arg for arg in cmd)
        assert {name: env[name] for name in secrets} == secrets

    def test_only_set_passthrough_vars_forwarded(self, monkeypatch) -> None:
        for name in DEFAULT_ENV_PASSTHROUGH:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "gh-token")
        monkeypatch.setenv("S3_ENDPOINT_URL", "")
        _, cmd, _ = self._launch()
        assert _forwarded_env_names(cmd) == ["TASK_ENVELOPE_JSON", "CUSTOMER_ID", "GITHUB_TOKEN"]

    def test_uses_configured_image_network_and_mount(self, monkeypatch) -> None:
        for name in DEFAULT_ENV_PASSTHROUGH:
            monkeypatch.delenv(name, raising=False)
        launcher = DockerLauncher(image="weyucou/agent-worker:local", network="net", aws_config_dir="/cfg")
        _, cmd, _ = self._launch(launcher=launcher)
        expected_names = ["TASK_ENVELOPE_JSON", "CUSTOMER_ID"]
        assert cmd == build_docker_command("weyucou/agent-worker:local", expected_names, "net", "/cfg")

    def test_docker_start_failure_is_failure(self) -> None:
        launcher = DockerLauncher(image="img")
        too_big = OSError(7, "Argument list too long")  # E2BIG: env var over the 128 KiB per-string limit
        with patch("marvin.launchers.docker.subprocess.Popen", side_effect=too_big):
            result = launcher.launch(TaskEnvelope.model_validate_json(self.RAW), self.RAW, lambda: None)
        assert not result.success
        assert "Argument list too long" in result.detail

    def test_heartbeat_while_container_runs(self) -> None:
        beats: list[int] = []
        launcher = DockerLauncher(image="img", heartbeat_seconds=7)
        timeout = subprocess.TimeoutExpired("docker", 7)
        with patch("marvin.launchers.docker.subprocess.Popen") as mock_popen:
            mock_popen.return_value.wait.side_effect = [timeout, timeout, 0]
            result = launcher.launch(TaskEnvelope.model_validate_json(self.RAW), self.RAW, lambda: beats.append(1))
        assert result.success
        assert len(beats) == 2
        mock_popen.return_value.wait.assert_called_with(timeout=7)


class TestEcsLauncherLaunch:
    RAW = _envelope_json()

    def _launch(self, launcher: EcsLauncher) -> LaunchResult:
        def no_heartbeat() -> None:
            pytest.fail("run_task returns immediately; the ECS backend must not heartbeat")

        return launcher.launch(TaskEnvelope.model_validate_json(self.RAW), self.RAW, no_heartbeat)

    def _describe_task(self, ecs_cluster: dict[str, Any], task_arn: str) -> dict[str, Any]:
        return ecs_cluster["ecs"].describe_tasks(cluster=ecs_cluster["cluster_arn"], tasks=[task_arn])["tasks"][0]

    def test_run_task_with_envelope_override(self, ecs_cluster) -> None:
        result = self._launch(ecs_cluster["launcher"])
        assert result.success
        task = self._describe_task(ecs_cluster, result.detail)
        assert task["launchType"] == "FARGATE"
        assert task["taskDefinitionArn"] == ecs_cluster["task_definition_arn"]
        assert task["overrides"]["containerOverrides"] == [
            {
                "name": "agent-worker",
                "environment": [
                    {"name": "TASK_ENVELOPE_JSON", "value": self.RAW},
                    {"name": "CUSTOMER_ID", "value": "cust-001"},
                ],
            }
        ]

    def test_task_network_uses_configured_subnet_and_security_group(self, ecs_cluster) -> None:
        result = self._launch(ecs_cluster["launcher"])
        attachment = self._describe_task(ecs_cluster, result.detail)["attachments"][0]
        eni_id = next(d["value"] for d in attachment["details"] if d["name"] == "networkInterfaceId")
        eni = ecs_cluster["ec2"].describe_network_interfaces(NetworkInterfaceIds=[eni_id])["NetworkInterfaces"][0]
        assert eni["SubnetId"] == ecs_cluster["subnet_id"]
        assert [group["GroupId"] for group in eni["Groups"]] == [ecs_cluster["group_id"]]

    def test_client_error_is_failure(self, ecs_cluster) -> None:
        launcher = ecs_cluster["launcher"]
        launcher.cluster = "missing-cluster"
        result = self._launch(launcher)
        assert not result.success
        assert "ClusterNotFoundException" in result.detail


class TestResultFromRunTask:
    def test_accepted_task(self) -> None:
        task_arn = "arn:aws:ecs:us-east-1:123456789012:task/agent-worker/abc123"
        result = result_from_run_task({"tasks": [{"taskArn": task_arn}], "failures": []})
        assert result == LaunchResult(success=True, detail=task_arn)

    def test_failures_mean_not_accepted(self) -> None:
        failure = {"arn": "arn:aws:ecs:us-east-1:123456789012:container-instance/x", "reason": "RESOURCE:MEMORY"}
        result = result_from_run_task({"tasks": [], "failures": [failure]})
        assert not result.success
        assert "RESOURCE:MEMORY" in result.detail

    def test_no_task_means_not_accepted(self) -> None:
        assert not result_from_run_task({"tasks": [], "failures": []}).success


class TestPollOnce:
    def _poll(self, sqs_queue: tuple[Any, str], launcher: FakeLauncher, max_messages: int = 1) -> None:
        sqs, queue_url = sqs_queue
        poll_once(sqs, launcher, queue_url, visibility_timeout=30, max_messages=max_messages, wait_time_seconds=0)

    def test_success_deletes_message(self, sqs_queue) -> None:
        sqs, queue_url = sqs_queue
        body = _envelope_json()
        sqs.send_message(QueueUrl=queue_url, MessageBody=body)
        launcher = FakeLauncher(heartbeat=True)
        self._poll(sqs_queue, launcher)
        assert launcher.launched == [body]  # forwarded unchanged
        assert _message_counts(sqs, queue_url) == (0, 0)

    def test_failure_retains_message_for_redelivery(self, sqs_queue) -> None:
        sqs, queue_url = sqs_queue
        sqs.send_message(QueueUrl=queue_url, MessageBody=_envelope_json())
        launcher = FakeLauncher(failing=frozenset({"task-001"}))
        self._poll(sqs_queue, launcher)
        assert len(launcher.launched) == 1
        assert _message_counts(sqs, queue_url) == (0, 1)  # in flight until the visibility timeout expires

    def test_each_message_decided_independently(self, sqs_queue) -> None:
        sqs, queue_url = sqs_queue
        for task_id in ("task-ok", "task-bad"):
            sqs.send_message(QueueUrl=queue_url, MessageBody=_envelope_json(task_id=task_id))
        launcher = FakeLauncher(failing=frozenset({"task-bad"}))
        self._poll(sqs_queue, launcher, max_messages=2)
        assert len(launcher.launched) == 2
        assert _message_counts(sqs, queue_url) == (0, 1)

    @pytest.mark.parametrize(
        "body",
        ["not-json{{{", json.dumps({"task_id": "x"}), PRE_CONTRACT_ENVELOPE_JSON],
        ids=["malformed-json", "missing-fields", "pre-contract-sender"],
    )
    def test_invalid_envelope_not_launched_and_retained(self, sqs_queue, body) -> None:
        sqs, queue_url = sqs_queue
        sqs.send_message(QueueUrl=queue_url, MessageBody=body)
        launcher = FakeLauncher()
        self._poll(sqs_queue, launcher)
        assert launcher.launched == []
        assert _message_counts(sqs, queue_url) == (0, 1)  # redelivered, then dead-lettered by the redrive policy

    def test_empty_queue_is_noop(self, sqs_queue) -> None:
        launcher = FakeLauncher()
        self._poll(sqs_queue, launcher)
        assert launcher.launched == []

    def test_delete_failure_after_success_is_logged_and_raised(self, sqs_queue) -> None:
        sqs, queue_url = sqs_queue
        sqs.send_message(QueueUrl=queue_url, MessageBody=_envelope_json())
        # The queue disappears while the task runs, so delete_message fails after a successful launch.
        launcher = FakeLauncher(on_launch=lambda: sqs.delete_queue(QueueUrl=queue_url))
        with capture_logs() as logs, pytest.raises(ClientError):
            self._poll(sqs_queue, launcher)
        errors = [entry for entry in logs if entry["log_level"] == "error"]
        assert len(errors) == 1
        assert errors[0]["event"].startswith("Task succeeded but delete_message failed")
        assert errors[0]["detail"] == "exit=0"
        assert errors[0]["error"]


class TestExtendVisibility:
    def test_sets_visibility_of_received_message(self, sqs_queue) -> None:
        sqs, queue_url = sqs_queue
        sqs.send_message(QueueUrl=queue_url, MessageBody="body")
        message = sqs.receive_message(QueueUrl=queue_url, VisibilityTimeout=300)["Messages"][0]
        assert _message_counts(sqs, queue_url) == (0, 1)
        # Timeout 0 makes the change observable: the message becomes visible again.
        extend_visibility(sqs, queue_url, message["ReceiptHandle"], visibility_timeout=0)
        assert _message_counts(sqs, queue_url) == (1, 0)

    def test_sqs_error_does_not_raise(self, sqs_queue) -> None:
        sqs, queue_url = sqs_queue
        sqs.send_message(QueueUrl=queue_url, MessageBody="body")
        message = sqs.receive_message(QueueUrl=queue_url)["Messages"][0]
        sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])
        extend_visibility(sqs, queue_url, message["ReceiptHandle"], visibility_timeout=300)


class TestMain:
    QUEUE_URL = "http://localhost:4566/000000000000/tasks"

    @pytest.fixture(autouse=True)
    def _restore_signal_handlers(self) -> Iterator[None]:
        handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
        yield
        for signum, handler in handlers.items():
            signal.signal(signum, handler)

    def test_missing_queue_url_exits_2(self, monkeypatch) -> None:
        monkeypatch.delenv("SQS_QUEUE_URL", raising=False)
        assert main() == 2

    def test_invalid_backend_exits_2(self, monkeypatch) -> None:
        monkeypatch.setenv("SQS_QUEUE_URL", self.QUEUE_URL)
        monkeypatch.setenv("COMPUTE_BACKEND", "lambda")
        assert main() == 2

    @pytest.mark.parametrize("name", ["VISIBILITY_TIMEOUT", "MAX_MESSAGES", "POLL_INTERVAL_SECONDS"])
    def test_non_numeric_setting_exits_2(self, monkeypatch, name) -> None:
        monkeypatch.setenv("SQS_QUEUE_URL", self.QUEUE_URL)
        monkeypatch.setenv(name, "five")
        assert main() == 2

    def test_retries_sqs_errors_and_stops_on_sigterm(self, monkeypatch, sqs_queue) -> None:
        _, queue_url = sqs_queue
        monkeypatch.setenv("SQS_QUEUE_URL", queue_url)
        monkeypatch.setenv("POLL_INTERVAL_SECONDS", "0")
        launcher = FakeLauncher()
        polls: list[tuple[Any, ...]] = []

        def poll(*args: Any, **_kwargs: Any) -> None:
            polls.append(args)
            if len(polls) == 1:
                raise ClientError({"Error": {"Code": "QueueDoesNotExist", "Message": "not yet"}}, "ReceiveMessage")
            signal.raise_signal(signal.SIGTERM)

        with (
            patch("marvin.launchers.consumer.get_launcher", return_value=launcher),
            patch("marvin.launchers.consumer.poll_once", side_effect=poll),
        ):
            assert main() == 0
        assert len(polls) == 2  # the SQS error is retried; SIGTERM stops after the current poll
        assert all(args[1] is launcher and args[2] == queue_url for args in polls)

    @pytest.mark.parametrize(
        ("launcher", "max_messages", "warned"),
        [
            (DockerLauncher(image="img"), "1", False),
            (DockerLauncher(image="img"), "2", True),
            (
                EcsLauncher(client=None, cluster="c", task_definition="t", subnets=["s"], security_groups=["g"]),
                "10",
                False,
            ),
        ],
        ids=["docker-single", "docker-batch", "ecs-batch"],
    )
    def test_warns_when_docker_receives_batches(
        self, monkeypatch, capsys, moto_region, launcher, max_messages, warned
    ) -> None:
        monkeypatch.setenv("SQS_QUEUE_URL", self.QUEUE_URL)
        monkeypatch.setenv("MAX_MESSAGES", max_messages)
        with (
            patch("marvin.launchers.consumer.get_launcher", return_value=launcher),
            patch(
                "marvin.launchers.consumer.poll_once",
                side_effect=lambda *_, **__: signal.raise_signal(signal.SIGTERM),
            ),
        ):
            assert main() == 0
        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        warnings = [line for line in lines if line["level"] == "warning"]
        assert [line["max_messages"] for line in warnings] == ([int(max_messages)] if warned else [])

    def test_logs_are_json_with_required_fields_and_redaction(self, monkeypatch, capsys) -> None:
        monkeypatch.delenv("SQS_QUEUE_URL", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "test")
        assert main() == 2  # emits one critical line through the launcher's logging setup
        sensitive = dict.fromkeys(("password", "token", "secret", "api_key"), "must-not-be-logged")
        structlog.get_logger().info("probe", **sensitive)
        lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert len(lines) == 2
        required = {"timestamp", "level", "event", "logger", "service", "environment"}
        assert all(required <= line.keys() for line in lines)
        assert lines[0]["level"] == "critical"
        assert lines[0]["logger"] == "marvin.launchers.consumer"
        assert lines[0]["service"] == "agent-worker-launcher"
        assert lines[0]["environment"] == "test"
        assert {lines[1][key] for key in sensitive} == {"[REDACTED]"}
