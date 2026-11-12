# Copyright (c) 2026 MNKT
"""Compute backends that run one agent-worker container per TaskEnvelope.

Select the backend with COMPUTE_BACKEND=docker|ecs. The SQS loop lives in consumer.py
(`python -m marvin.launchers`).
"""

import os

from marvin.definitions import StringEnumWithChoices
from marvin.launchers.base import Launcher, LaunchResult
from marvin.launchers.docker import DockerLauncher
from marvin.launchers.ecs import EcsLauncher


class ComputeBackend(StringEnumWithChoices):
    DOCKER = "docker"
    ECS = "ecs"


def get_launcher(backend: str | None = None) -> Launcher:
    """Build the launcher for `backend`, else COMPUTE_BACKEND, else docker.

    Raises ValueError on an unknown backend or missing backend config, so startup fails fast.
    """
    value = os.getenv("COMPUTE_BACKEND", ComputeBackend.DOCKER.value) if backend is None else backend
    try:
        selected = ComputeBackend(value)
    except ValueError:
        valid = ", ".join(ComputeBackend.values())
        raise ValueError(f"Invalid COMPUTE_BACKEND={value!r}; expected one of: {valid}") from None
    if selected is ComputeBackend.ECS:
        return EcsLauncher.from_env()
    return DockerLauncher.from_env()


__all__ = ["ComputeBackend", "DockerLauncher", "EcsLauncher", "LaunchResult", "Launcher", "get_launcher"]
