# Copyright (c) 2026 MNKT
"""Launcher interface shared by the compute backends."""

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from marvin.models import TaskEnvelope

if TYPE_CHECKING:
    from collections.abc import Callable


class LaunchResult(BaseModel):
    """Outcome of one launch. The consumer deletes the SQS message only when `success` is true."""

    success: bool
    detail: str  # exit code (docker) or task ARN / error (ecs), for logs


class Launcher(Protocol):
    """Runs one envelope on a compute backend.

    raw_json is the original SQS message body, passed unchanged as TASK_ENVELOPE_JSON.
    heartbeat extends the message's SQS visibility; backends that block call it periodically.
    """

    def launch(self, envelope: TaskEnvelope, raw_json: str, heartbeat: Callable[[], None]) -> LaunchResult: ...
