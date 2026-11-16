# Copyright (c) 2026 MNKT
"""Docker backend: `docker run --rm` the agent-worker image, the local equivalent of ecs:RunTask."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Self

from marvin.launchers.base import LaunchResult
from marvin.models import TaskEnvelope

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

DEFAULT_IMAGE = "weyucou/agent-worker:latest"
CONTAINER_AWS_CONFIG_DIR = "/root/.aws"  # the agent-worker image runs as root

# Launcher env vars forwarded to the container when set (non-empty).
# AWS_BEARER_TOKEN_BEDROCK is left out on purpose: in the container it would replace IAM auth for Bedrock.
DEFAULT_ENV_PASSTHROUGH = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_REGION",
    "BEDROCK_AWS_REGION",
    "BEDROCK_AWS_PROFILE",
    "S3_ENDPOINT_URL",
    "SECRETS_MANAGER_ENDPOINT_URL",
    "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN",
    "ENVIRONMENT",
)


def build_docker_command(
    image: str, env_names: Iterable[str], network: str = "", aws_config_dir: str = ""
) -> list[str]:
    """Return the `docker run` argv.

    Env vars are passed by name only (`-e NAME`): docker reads each value from the
    calling process env, so secrets and the envelope never appear in argv or `ps`.
    """
    cmd = ["docker", "run", "--rm"]
    if network:
        cmd += ["--network", network]
    if aws_config_dir:
        cmd += ["-v", f"{aws_config_dir}:{CONTAINER_AWS_CONFIG_DIR}:ro"]
    for name in env_names:
        cmd += ["-e", name]
    cmd.append(image)
    return cmd


class DockerLauncher:
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        env_passthrough: Iterable[str] = DEFAULT_ENV_PASSTHROUGH,
        network: str = "",
        aws_config_dir: str = "",
        heartbeat_seconds: float = 150,
    ) -> None:
        self.image = image
        self.env_passthrough = tuple(env_passthrough)
        self.network = network
        self.aws_config_dir = aws_config_dir
        self.heartbeat_seconds = heartbeat_seconds

    @classmethod
    def from_env(cls) -> Self:
        if shutil.which("docker") is None:
            raise ValueError("docker CLI not found on PATH; COMPUTE_BACKEND=docker requires it")
        extra = [name.strip() for name in os.getenv("DOCKER_ENV_PASSTHROUGH", "").split(",") if name.strip()]
        aws_config_dir = os.getenv("DOCKER_AWS_CONFIG_DIR", "")
        # Same variable the consumer uses: extend visibility well before it expires.
        visibility_timeout = int(os.getenv("VISIBILITY_TIMEOUT", "300"))
        return cls(
            image=os.getenv("AGENT_WORKER_IMAGE", DEFAULT_IMAGE),
            env_passthrough=(*DEFAULT_ENV_PASSTHROUGH, *extra),
            network=os.getenv("DOCKER_NETWORK", ""),
            aws_config_dir=str(Path(aws_config_dir).expanduser()) if aws_config_dir else "",
            heartbeat_seconds=max(1, visibility_timeout // 2),
        )

    def launch(self, envelope: TaskEnvelope, raw_json: str, heartbeat: Callable[[], None]) -> LaunchResult:
        """Run one container and block until it exits; success means exit code 0."""
        env = {**os.environ, "TASK_ENVELOPE_JSON": raw_json, "CUSTOMER_ID": envelope.customer_id}
        passthrough = [name for name in self.env_passthrough if env.get(name)]
        cmd = build_docker_command(
            self.image, ["TASK_ENVELOPE_JSON", "CUSTOMER_ID", *passthrough], self.network, self.aws_config_dir
        )
        # Container output is not captured: entrypoint.py's JSON logs go straight to our stdout.
        try:
            proc = subprocess.Popen(cmd, env=env)
        except OSError as exc:  # e.g. E2BIG: the envelope env var exceeds the 128 KiB per-string limit
            return LaunchResult(success=False, detail=f"docker run failed to start: {exc}")
        returncode = None
        while returncode is None:
            try:
                returncode = proc.wait(timeout=self.heartbeat_seconds)
            except subprocess.TimeoutExpired:
                heartbeat()
        return LaunchResult(success=returncode == 0, detail=f"exit={returncode}")
