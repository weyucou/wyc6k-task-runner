"""SQS loop: receive a TaskEnvelope, launch it, delete the message only on success.

Run with: python -m marvin.launchers
"""

import functools
import os
import signal
import threading
import time
from typing import Any

import boto3
import structlog
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import ValidationError

from marvin.launchers import DockerLauncher, get_launcher
from marvin.launchers.base import Launcher
from marvin.models import TaskEnvelope

SERVICE_NAME = "agent-worker-launcher"
SQS_WAIT_TIME_SECONDS = 20

_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "token",
        "secret",
        "ssn",
        "email",
        "credit_card",
        "authorization",
        "api_key",
    }
)

log = structlog.get_logger()


def _scrub_sensitive_fields(
    _logger: structlog.types.WrappedLogger,
    _method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    for key in _SENSITIVE_KEYS & event_dict.keys():
        event_dict[key] = "[REDACTED]"
    return event_dict


def configure_logging() -> None:
    """JSON logs to stdout; same processors as entrypoint.py."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _scrub_sensitive_fields,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
        # A few lines per task: without the cache, the module-level logger always uses the current config.
        cache_logger_on_first_use=False,
    )


def get_sqs_client() -> Any:
    kwargs: dict[str, Any] = {"region_name": os.getenv("AWS_DEFAULT_REGION", "ap-northeast-1")}
    if endpoint_url := os.getenv("SQS_ENDPOINT_URL"):
        kwargs["endpoint_url"] = endpoint_url
    return boto3.client("sqs", **kwargs)


def extend_visibility(sqs: Any, queue_url: str, receipt_handle: str, visibility_timeout: int) -> None:
    """Heartbeat: keep a running task's message hidden so it is not launched twice."""
    try:
        sqs.change_message_visibility(
            QueueUrl=queue_url, ReceiptHandle=receipt_handle, VisibilityTimeout=visibility_timeout
        )
    except (BotoCoreError, ClientError) as exc:
        log.warning("Failed to extend message visibility", error=str(exc))


def _parse_envelope(body: str) -> TaskEnvelope | None:
    try:
        return TaskEnvelope.model_validate_json(body)
    except ValidationError as exc:
        # Inputs stay out of the log: an envelope may carry secrets such as agent.api_key.
        errors = exc.errors(include_url=False, include_input=False)
    log.error("Invalid task envelope; message left for redelivery / DLQ", errors=errors)
    return None


def _process_message(
    sqs: Any, launcher: Launcher, queue_url: str, message: dict[str, Any], visibility_timeout: int
) -> None:
    body = message["Body"]
    receipt_handle = message["ReceiptHandle"]
    envelope = _parse_envelope(body)
    if envelope is None:
        return

    with structlog.contextvars.bound_contextvars(task_id=envelope.task_id, customer_id=envelope.customer_id):
        log.info("Launching task")
        heartbeat = functools.partial(extend_visibility, sqs, queue_url, receipt_handle, visibility_timeout)
        result = launcher.launch(envelope, body, heartbeat)
        if not result.success:
            log.error("Task failed; message left for redelivery", detail=result.detail)
            return
        try:
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
        except (BotoCoreError, ClientError) as exc:
            delete_error = exc
        else:
            log.info("Task launched; message deleted", detail=result.detail)
            return
        # At-least-once: the message reappears once its visibility expires, and the task runs again.
        log.error(
            "Task succeeded but delete_message failed; it will run again",
            detail=result.detail,
            error=str(delete_error),
        )
        raise delete_error


def poll_once(
    sqs: Any,
    launcher: Launcher,
    queue_url: str,
    visibility_timeout: int,
    max_messages: int = 1,
    wait_time_seconds: int = SQS_WAIT_TIME_SECONDS,
) -> None:
    """Receive up to `max_messages` envelopes and launch them one at a time."""
    response = sqs.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=max_messages,
        WaitTimeSeconds=wait_time_seconds,
        VisibilityTimeout=visibility_timeout,
    )
    for message in response.get("Messages", []):
        with structlog.contextvars.bound_contextvars(message_id=message["MessageId"]):
            _process_message(sqs, launcher, queue_url, message, visibility_timeout)


def main() -> int:
    """Poll until SIGTERM/SIGINT. Returns 2 on invalid config (as entrypoint.py does), else 0."""
    configure_logging()
    structlog.contextvars.bind_contextvars(
        service=SERVICE_NAME, environment=os.getenv("ENVIRONMENT", "production"), logger=__name__
    )

    queue_url = os.getenv("SQS_QUEUE_URL", "")
    if not queue_url:
        log.critical("SQS_QUEUE_URL is not set")
        return 2
    try:
        visibility_timeout = int(os.getenv("VISIBILITY_TIMEOUT", "300"))
        max_messages = int(os.getenv("MAX_MESSAGES", "1"))
        poll_interval = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))
        launcher = get_launcher()
    except ValueError as exc:
        log.critical("Invalid launcher configuration", error=str(exc))
        return 2
    if max_messages > 1 and isinstance(launcher, DockerLauncher):
        # Only the running message is heartbeated; the rest of a batch can reappear while it waits.
        log.warning(
            "MAX_MESSAGES > 1 with the docker backend: waiting messages are not heartbeated and can run twice",
            max_messages=max_messages,
        )

    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda _signum, _frame: stop.set())

    sqs = get_sqs_client()
    log.info("Launcher started", launcher=type(launcher).__name__, queue_url=queue_url)
    # Checked between polls: the launcher waits for a running container and never stops it. A terminal
    # Ctrl-C still reaches the container, since the `docker run` child shares the foreground process group.
    while not stop.is_set():
        try:
            poll_once(sqs, launcher, queue_url, visibility_timeout, max_messages)
        except (BotoCoreError, ClientError) as exc:
            log.warning("SQS call failed; retrying", error=str(exc), retry_in_seconds=poll_interval)
            # time.sleep, not stop.wait: Event.set() from a signal handler can deadlock a waiting main thread.
            time.sleep(poll_interval)
    log.info("Launcher stopped")
    return 0
