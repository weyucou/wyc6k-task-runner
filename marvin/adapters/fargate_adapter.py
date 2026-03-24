import os

import boto3

from marvin.adapters.base import BaseComputeAdapter, DispatchResult
from marvin.definitions import AWS_REGION
from marvin.models import TaskEnvelope

FARGATE_TASK_DEFINITION_ARN = os.getenv("FARGATE_TASK_DEFINITION_ARN", "")
FARGATE_CLUSTER = os.getenv("FARGATE_CLUSTER", "")


class FargateAdapter(BaseComputeAdapter):
    def __init__(
        self,
        task_definition_arn: str | None = None,
        cluster: str | None = None,
    ) -> None:
        self.task_definition_arn = task_definition_arn or FARGATE_TASK_DEFINITION_ARN
        self.cluster = cluster or FARGATE_CLUSTER

    async def dispatch(self, envelope: TaskEnvelope) -> DispatchResult:
        client = boto3.client("ecs", region_name=AWS_REGION)
        envelope_json = envelope.model_dump_json()
        response = client.run_task(
            taskDefinition=self.task_definition_arn,
            cluster=self.cluster,
            overrides={
                "containerOverrides": [
                    {
                        "name": "worker",
                        "environment": [
                            {"name": "TASK_ENVELOPE", "value": envelope_json},
                        ],
                    },
                ],
            },
            launchType="FARGATE",
        )
        tasks = response.get("tasks", [])
        if not tasks:
            return DispatchResult(adapter="fargate", job_id=None, status="failed")
        task_arn: str = tasks[0]["taskArn"]
        job_id = task_arn.split("/")[-1]
        return DispatchResult(adapter="fargate", job_id=job_id, status="dispatched")
