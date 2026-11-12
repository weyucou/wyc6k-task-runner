# Copyright (c) 2026 MNKT
"""ECS backend: ecs:RunTask on Fargate with the TASK_ENVELOPE_JSON container override."""

import os
from typing import TYPE_CHECKING, Any, Self

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from marvin.launchers.base import LaunchResult
from marvin.models import TaskEnvelope

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_CONTAINER_NAME = "agent-worker"  # container name in wyc6k-infra stacks/ecs-agent-worker.yaml


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def result_from_run_task(response: dict[str, Any]) -> LaunchResult:
    """Success means ECS accepted the task (one task, no failures); detail is the task ARN."""
    tasks = response.get("tasks", [])
    failures = response.get("failures", [])
    if failures or not tasks:
        return LaunchResult(success=False, detail=f"run_task failures: {failures}")
    return LaunchResult(success=True, detail=tasks[0]["taskArn"])


class EcsLauncher:
    def __init__(
        self,
        client: Any,
        cluster: str,
        task_definition: str,
        subnets: list[str],
        security_groups: list[str],
        *,
        assign_public_ip: str = "DISABLED",
        container_name: str = DEFAULT_CONTAINER_NAME,
    ) -> None:
        self.client = client
        self.cluster = cluster
        self.task_definition = task_definition
        self.subnets = subnets
        self.security_groups = security_groups
        self.assign_public_ip = assign_public_ip
        self.container_name = container_name

    @classmethod
    def from_env(cls) -> Self:
        cluster = os.getenv("ECS_CLUSTER", "").strip()
        task_definition = os.getenv("ECS_TASK_DEFINITION", "").strip()
        subnets = _split_csv(os.getenv("ECS_SUBNETS", ""))
        security_groups = _split_csv(os.getenv("ECS_SECURITY_GROUPS", ""))
        required = {
            "ECS_CLUSTER": cluster,
            "ECS_TASK_DEFINITION": task_definition,
            "ECS_SUBNETS": subnets,
            "ECS_SECURITY_GROUPS": security_groups,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"COMPUTE_BACKEND=ecs requires {', '.join(missing)}")
        return cls(
            client=boto3.client("ecs", region_name=os.getenv("AWS_DEFAULT_REGION", "ap-northeast-1")),
            cluster=cluster,
            task_definition=task_definition,
            subnets=subnets,
            security_groups=security_groups,
            assign_public_ip=os.getenv("ECS_ASSIGN_PUBLIC_IP", "DISABLED"),
            container_name=os.getenv("ECS_CONTAINER_NAME", DEFAULT_CONTAINER_NAME),
        )

    def launch(self, envelope: TaskEnvelope, raw_json: str, heartbeat: Callable[[], None]) -> LaunchResult:
        """Dispatch one Fargate task and return once ECS accepts it.

        run_task is asynchronous, so the container exit code is not observed here; failures after
        dispatch surface through ECS task state / CloudWatch, not SQS redelivery.
        """
        try:
            response = self.client.run_task(
                cluster=self.cluster,
                taskDefinition=self.task_definition,
                launchType="FARGATE",
                count=1,
                networkConfiguration={
                    "awsvpcConfiguration": {
                        "subnets": self.subnets,
                        "securityGroups": self.security_groups,
                        "assignPublicIp": self.assign_public_ip,
                    }
                },
                overrides={
                    "containerOverrides": [
                        {
                            "name": self.container_name,
                            "environment": [
                                {"name": "TASK_ENVELOPE_JSON", "value": raw_json},
                                {"name": "CUSTOMER_ID", "value": envelope.customer_id},
                            ],
                        }
                    ]
                },
            )
        except (BotoCoreError, ClientError) as exc:
            return LaunchResult(success=False, detail=f"run_task error: {exc}")
        return result_from_run_task(response)
