# Copyright (c) 2026 MNKT
import os

AWS_DEFAULT_REGION: str = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-1")
SECRETS_MANAGER_ENDPOINT_URL: str = os.getenv("SECRETS_MANAGER_ENDPOINT_URL", "")
ZAATAR_SEARCH_API_URL: str = os.getenv("ZAATAR_SEARCH_API_URL", "")
# Memory given to the model: the newest memory files read from S3, and the character budget
# for MEMORY.md plus those entries in the system prompt.
MEMORY_RECENT_ENTRIES: int = int(os.getenv("MEMORY_RECENT_ENTRIES", "7"))
MEMORY_CONTEXT_MAX_CHARS: int = int(os.getenv("MEMORY_CONTEXT_MAX_CHARS", "20000"))
