import os

import boto3

from marvin.adapters.base import BaseComputeAdapter, DispatchResult
from marvin.definitions import AWS_REGION
from marvin.models import TaskEnvelope

BATCH_JOB_DEFINITION = os.getenv("BATCH_JOB_DEFINITION", "")
BATCH_JOB_QUEUE = os.getenv("BATCH_JOB_QUEUE", "")


class BatchAdapter(BaseComputeAdapter):
    def __init__(
        self,
        job_definition: str | None = None,
        job_queue: str | None = None,
    ) -> None:
        self.job_definition = job_definition or BATCH_JOB_DEFINITION
        self.job_queue = job_queue or BATCH_JOB_QUEUE

    async def dispatch(self, envelope: TaskEnvelope) -> DispatchResult:
        client = boto3.client("batch", region_name=AWS_REGION)
        envelope_json = envelope.model_dump_json()
        response = client.submit_job(
            jobName=f"marvin-task-{envelope.task_id}",
            jobDefinition=self.job_definition,
            jobQueue=self.job_queue,
            containerOverrides={
                "environment": [
                    {"name": "TASK_ENVELOPE", "value": envelope_json},
                ],
            },
        )
        job_id: str | None = response.get("jobId")
        status = "dispatched" if job_id else "failed"
        return DispatchResult(adapter="batch", job_id=job_id, status=status)
