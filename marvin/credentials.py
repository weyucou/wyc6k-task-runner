"""AWS Secrets Manager credential resolver for the marvin worker."""

import json
import logging
import os
from typing import Any

import boto3

from marvin.models import TaskEnvelope

logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-1")
SECRETS_MANAGER_ENDPOINT_URL = os.getenv("SECRETS_MANAGER_ENDPOINT_URL", "")


def _get_secrets_client() -> Any:
    kwargs: dict[str, Any] = {"region_name": AWS_REGION}
    if SECRETS_MANAGER_ENDPOINT_URL:
        kwargs["endpoint_url"] = SECRETS_MANAGER_ENDPOINT_URL
    return boto3.client("secretsmanager", **kwargs)


def _fetch_secret(arn: str) -> str:
    """Fetch a secret value from AWS Secrets Manager by ARN.

    Supports both plain string secrets and JSON objects with a "value" key.
    Raises RuntimeError on failure so the worker aborts before processing.
    """
    client = _get_secrets_client()
    try:
        response = client.get_secret_value(SecretId=arn)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch secret ARN={arn}: {exc}") from exc

    raw = response.get("SecretString", "")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "value" in parsed:
            return parsed["value"]
    except (json.JSONDecodeError, TypeError):
        pass
    return raw


def _resolve_credential(arn: str | None, env_var: str, label: str) -> str | None:
    """Fetch a credential from Secrets Manager if an ARN is given, else from env.

    Returns None when neither source has a value.
    """
    if arn:
        logger.info("Fetching %s from Secrets Manager", label)
        return _fetch_secret(arn)
    value = os.getenv(env_var)
    if value:
        logger.info("Using %s from environment", env_var)
    return value


class CredentialResolver:
    """Resolves customer credentials from AWS Secrets Manager or environment variables.

    Never logs resolved secret values.
    """

    def resolve(self, envelope: TaskEnvelope) -> None:
        """Resolve credentials and apply them to the task envelope in-place.

        - github_token_arn -> sets os.environ["GITHUB_TOKEN"] for tool use
        - anthropic_api_key_arn -> sets envelope.agent.api_key for the LLM client

        Falls back to GITHUB_TOKEN / ANTHROPIC_API_KEY env vars when ARNs are absent.
        Raises RuntimeError before processing if an ARN fetch fails.
        """
        github_token = _resolve_credential(envelope.github_token_arn, "GITHUB_TOKEN", "GitHub token")
        if github_token:
            os.environ["GITHUB_TOKEN"] = github_token

        anthropic_key = _resolve_credential(envelope.anthropic_api_key_arn, "ANTHROPIC_API_KEY", "Anthropic API key")
        if anthropic_key:
            envelope.agent.api_key = anthropic_key
