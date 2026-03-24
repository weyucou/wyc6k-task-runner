from marvin.adapters.base import BaseComputeAdapter
from marvin.adapters.batch_adapter import BatchAdapter
from marvin.adapters.fargate_adapter import FargateAdapter
from marvin.adapters.lambda_adapter import LambdaAdapter
from marvin.models import TaskEnvelope

LAMBDA_MAX_SECONDS = 600
FARGATE_MAX_SECONDS = 21600


class AdapterFactory:
    @staticmethod
    def select(envelope: TaskEnvelope) -> BaseComputeAdapter:
        if envelope.duration_hint_seconds < LAMBDA_MAX_SECONDS:
            return LambdaAdapter()
        if envelope.duration_hint_seconds <= FARGATE_MAX_SECONDS:
            return FargateAdapter()
        return BatchAdapter()
