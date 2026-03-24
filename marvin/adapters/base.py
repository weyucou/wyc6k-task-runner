from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

from marvin.models import TaskEnvelope


class DispatchResult(BaseModel):
    adapter: str
    job_id: str | None
    status: Literal["dispatched", "inline_complete", "failed"]
    output: str | None = None


class BaseComputeAdapter(ABC):
    @abstractmethod
    async def dispatch(self, envelope: TaskEnvelope) -> DispatchResult: ...
