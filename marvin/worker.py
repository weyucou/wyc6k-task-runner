# Copyright (c) 2026 MNKT
"""SQS consumer worker entry point.

Run with: python -m marvin.worker
"""

import asyncio
import logging
import os
import signal
import threading
from typing import Any

import boto3
from pydantic import ValidationError

from marvin.context import ContextBundleService
from marvin.credentials import CredentialResolver
from marvin.llm import LLMMessage
from marvin.models import TaskEnvelope
from marvin.runner import AgentRunner
from marvin.tools.coding import stop_background_processes

logger = logging.getLogger(__name__)

SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL", "")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-1")
SQS_ENDPOINT_URL = os.getenv("SQS_ENDPOINT_URL", "")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "1"))
VISIBILITY_TIMEOUT = int(os.getenv("VISIBILITY_TIMEOUT", "300"))
WAIT_TIME_SECONDS = int(os.getenv("SQS_WAIT_TIME_SECONDS", "20"))

_shutdown = threading.Event()


def _handle_signal(signum: int, _frame: Any) -> None:
    logger.info("Received signal %d, shutting down...", signum)
    _shutdown.set()


def _get_sqs_client() -> Any:
    kwargs: dict[str, Any] = {"region_name": AWS_REGION}
    if SQS_ENDPOINT_URL:
        kwargs["endpoint_url"] = SQS_ENDPOINT_URL
    return boto3.client("sqs", **kwargs)


async def process_envelope(envelope: TaskEnvelope) -> dict[str, Any]:
    """Process a single TaskEnvelope through the agent runner."""
    context_service = ContextBundleService()
    context = context_service.pull(envelope.s3_context_prefix)
    logger.info("Pulled context for customer=%s", context.customer_id)

    messages: list[LLMMessage] = []
    for msg in envelope.conversation_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            messages.append(LLMMessage.system(content))
        elif role == "assistant":
            messages.append(LLMMessage.assistant(content))
        else:
            messages.append(LLMMessage.user(content))

    # The system prompt is the context bundle followed by agent.system_prompt, as in entrypoint.py.
    runner = AgentRunner(
        agent=envelope.agent,
        session_id=envelope.session_id,
        context_bundle=context,
    )
    try:
        response_text, history = await runner.chat(
            envelope.user_message,
            conversation_history=messages,
            enable_tools=envelope.enable_tools,
        )
    finally:
        stop_background_processes()  # the worker is long-lived: sessions must not outlive their task

    return {
        "task_id": envelope.task_id,
        "customer_id": envelope.customer_id,
        "session_id": envelope.session_id,
        "response": response_text,
        "history": [m.to_dict() for m in history],
    }


def _parse_envelope(body: str) -> TaskEnvelope | None:
    try:
        return TaskEnvelope.model_validate_json(body)
    except ValidationError as exc:
        # Inputs stay out of the log: an envelope may carry secrets such as agent.api_key.
        errors = exc.errors(include_url=False, include_input=False)
    logger.error("Invalid task envelope; message left for redelivery / DLQ: %s", errors)
    return None


def poll_once(sqs: Any) -> None:
    """Poll SQS for one batch of messages and process them."""
    if not SQS_QUEUE_URL:
        logger.error("SQS_QUEUE_URL is not set")
        return

    response = sqs.receive_message(
        QueueUrl=SQS_QUEUE_URL,
        MaxNumberOfMessages=MAX_MESSAGES,
        WaitTimeSeconds=WAIT_TIME_SECONDS,
        VisibilityTimeout=VISIBILITY_TIMEOUT,
    )

    messages = response.get("Messages", [])
    if not messages:
        return

    for message in messages:
        receipt_handle = message["ReceiptHandle"]
        envelope = _parse_envelope(message["Body"])
        if envelope is None:
            continue
        try:
            resolver = CredentialResolver()
            resolver.resolve(envelope)
            logger.info("Processing task_id=%s for customer=%s", envelope.task_id, envelope.customer_id)

            result = asyncio.run(process_envelope(envelope))
            logger.info("Completed task_id=%s: %d chars", result["task_id"], len(result["response"]))
            sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
        except Exception:
            logger.exception("Failed to process message")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} [{levelname:5}] ({name}) {funcName}: {message}",
        style="{",
    )
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("marvin-manager worker starting (queue=%s)", SQS_QUEUE_URL)
    sqs = _get_sqs_client()

    while not _shutdown.is_set():
        try:
            poll_once(sqs)
        except Exception:
            logger.exception("Poll error")

    logger.info("Worker shutdown complete")


if __name__ == "__main__":
    main()
