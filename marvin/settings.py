# Copyright (c) 2026 MNKT
import os


def _env_int(name: str, default: int, *, minimum: int) -> int:
    """An integer environment variable; a value that is not an integer, or below `minimum`, raises ValueError."""
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


AWS_DEFAULT_REGION: str = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-1")
SECRETS_MANAGER_ENDPOINT_URL: str = os.getenv("SECRETS_MANAGER_ENDPOINT_URL", "")
ZAATAR_SEARCH_API_URL: str = os.getenv("ZAATAR_SEARCH_API_URL", "")

# The tool output cap (marvin/tools/truncate.py): the most a tool result shows the model, notice included.
# The floors leave room for the output, the truncation notice and an error message.
TOOL_OUTPUT_MAX_LINES: int = _env_int("MARVIN_TOOL_OUTPUT_MAX_LINES", 2000, minimum=10)
TOOL_OUTPUT_MAX_BYTES: int = _env_int("MARVIN_TOOL_OUTPUT_MAX_BYTES", 51200, minimum=1024)  # UTF-8 bytes
