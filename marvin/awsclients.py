"""AWS client factory functions."""

from typing import Any

import boto3

from marvin.settings import AWS_DEFAULT_REGION, SECRETS_MANAGER_ENDPOINT_URL


def get_secrets_manager_client() -> Any:
    kwargs: dict[str, Any] = {"region_name": AWS_DEFAULT_REGION}
    if SECRETS_MANAGER_ENDPOINT_URL:
        kwargs["endpoint_url"] = SECRETS_MANAGER_ENDPOINT_URL
    return boto3.client("secretsmanager", **kwargs)
