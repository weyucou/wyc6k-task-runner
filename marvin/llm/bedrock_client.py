# Copyright (c) 2026 MNKT
"""Claude on Amazon Bedrock, through the Bedrock Mantle (Messages API) endpoint."""

import os
from typing import Any

import boto3
from anthropic.lib.bedrock import AsyncAnthropicBedrockMantle  # `anthropic` re-exports it via `import *` only

from marvin.llm.anthropic_client import AnthropicClient


class BedrockClient(AnthropicClient):
    """AnthropicClient that reaches Claude through Amazon Bedrock with AWS credentials.

    Requests and responses are the Anthropic ones; only the SDK client differs. `api_key` and
    `base_url` are never used. Settings: BEDROCK_AWS_REGION (falls back to AWS_REGION, then
    AWS_DEFAULT_REGION) and BEDROCK_AWS_PROFILE (falls back to the default AWS credential chain).
    """

    def _get_client(self) -> Any:
        """Get or create the Bedrock Mantle client; raise if it has no credentials to sign with."""
        if self._client is None:
            region = os.getenv("BEDROCK_AWS_REGION") or None
            profile = os.getenv("BEDROCK_AWS_PROFILE") or None
            client = AsyncAnthropicBedrockMantle(aws_region=region, aws_profile=profile)
            # The SDK resolves credentials per request, and generate() turns request errors into a reply,
            # so the task would exit 0. Check here, outside generate()'s try. `client.api_key` is None
            # unless a Bedrock API key (AWS_BEARER_TOKEN_BEDROCK) replaces SigV4 signing.
            if client.api_key is None and boto3.Session(profile_name=profile).get_credentials() is None:
                raise RuntimeError(f"No AWS credentials for Bedrock (profile: {profile or 'default credential chain'})")
            self._client = client
        return self._client
