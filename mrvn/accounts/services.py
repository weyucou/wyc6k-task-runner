from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import boto3

if TYPE_CHECKING:
    import uuid
from botocore.exceptions import ClientError

from accounts.models import CredentialService, CustomerCredential

logger = logging.getLogger(__name__)

_ENV_VAR_MAP: dict[str, str] = {
    CredentialService.github: "GITHUB_TOKEN",
    CredentialService.anthropic: "ANTHROPIC_API_KEY",
    CredentialService.gemini: "GEMINI_API_KEY",
}


class CustomerCredentialService:
    def get(self, customer_id: uuid.UUID, service: CredentialService) -> str:
        """Fetch secret plaintext from AWS Secrets Manager."""
        credential = CustomerCredential.objects.get(customer_id=customer_id, service=service)
        client = boto3.client("secretsmanager")
        response = client.get_secret_value(SecretId=credential.secret_arn)
        return response["SecretString"]

    def inject_env(self, customer_id: uuid.UUID) -> dict[str, str]:
        """Return env var dict for all customer credentials, ready for subprocess injection."""
        env: dict[str, str] = {}
        for credential in CustomerCredential.objects.filter(customer_id=customer_id):
            service = CredentialService(credential.service)
            env_var = _ENV_VAR_MAP.get(service)
            if env_var is None:
                logger.warning("No env var mapping for service %s", service)
                continue
            try:
                client = boto3.client("secretsmanager")
                response = client.get_secret_value(SecretId=credential.secret_arn)
                env[env_var] = response["SecretString"]
            except ClientError:
                logger.exception("Failed to fetch secret for service %s", service)
        return env
