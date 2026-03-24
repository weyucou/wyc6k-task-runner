import os
import secrets
import time
import uuid
from typing import Any

import boto3


def uuidv7() -> uuid.UUID:
    """Generate a UUIDv7.

    Packs the current millisecond timestamp into the first 6 bytes,
    sets UUID version 7 (0x70) and RFC variant (0x80), with the
    remaining bytes as cryptographic random.
    """
    value = bytearray(secrets.token_bytes(16))

    timestamp = int(time.time() * 1000)

    value[0] = (timestamp >> 40) & 0xFF
    value[1] = (timestamp >> 32) & 0xFF
    value[2] = (timestamp >> 24) & 0xFF
    value[3] = (timestamp >> 16) & 0xFF
    value[4] = (timestamp >> 8) & 0xFF
    value[5] = timestamp & 0xFF

    value[6] = (value[6] & 0x0F) | 0x70
    value[8] = (value[8] & 0x3F) | 0x80

    return uuid.UUID(bytes=bytes(value))


def get_s3_client() -> Any:
    """Return a boto3 S3 client.

    Reads AWS_ENDPOINT_URL from the environment when present, allowing
    LocalStack to be used in development and testing without code changes.
    """
    kwargs: dict[str, Any] = {}
    endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    return boto3.client("s3", **kwargs)
