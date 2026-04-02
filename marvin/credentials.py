"""AWS Secrets Manager credential resolver for the marvin worker."""

import json
import logging
import os

from marvin.awsclients import get_secrets_manager_client
from marvin.models import TaskEnvelope

logger = logging.getLogger(__name__)


def _fetch_secret(secret_id: str) -> str:
    """Fetch a secret value from AWS Secrets Manager by secret ID or ARN.

    Supports both plain string secrets and JSON objects with a "value" key.
    Raises RuntimeError on failure so the worker aborts before processing.
    """
    client = get_secrets_manager_client()
    try:
        response = client.get_secret_value(SecretId=secret_id)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch secret {secret_id}: {exc}") from exc

    raw = response.get("SecretString", "")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "value" in parsed:
            return parsed["value"]
    except (json.JSONDecodeError, TypeError):
        pass
    return raw


def _resolve_credential(secret_id: str | None, env_var: str, label: str) -> str | None:
    """Fetch a credential from Secrets Manager if a secret ID is given, else from env.

    Returns None when neither source has a value.
    """
    if secret_id:
        logger.info("Fetching %s from Secrets Manager", label)
        return _fetch_secret(secret_id)
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

        - github_token_secret_id -> sets os.environ["GITHUB_TOKEN"] for tool use
        - anthropic_api_key_secret_id -> sets envelope.agent.api_key for the LLM client

        Falls back to GITHUB_TOKEN / ANTHROPIC_API_KEY env vars when secret IDs are absent.
        Raises RuntimeError before processing if a Secrets Manager fetch fails.
        """
        github_token = _resolve_credential(envelope.github_token_secret_id, "GITHUB_TOKEN", "GitHub token")
        if github_token:
            os.environ["GITHUB_TOKEN"] = github_token

        anthropic_key = _resolve_credential(envelope.anthropic_api_key_secret_id, "ANTHROPIC_API_KEY", "Anthropic API key")
        if anthropic_key:
            envelope.agent.api_key = anthropic_key
