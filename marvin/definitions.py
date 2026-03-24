import os
from enum import Enum

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-1")

MAX_S3_BUCKET_LENGTH = 63
MAX_S3_KEY_LENGTH = 1024
SEP_AND_PREFIX_LENGTH = 5
MAX_STORAGE_PATH_LENGTH = SEP_AND_PREFIX_LENGTH + MAX_S3_BUCKET_LENGTH + MAX_S3_KEY_LENGTH


class IntegerEnumWithChoices(int, Enum):
    @classmethod
    def choices(cls) -> tuple[tuple[int, str], ...]:
        return tuple((e.value, str(e.value)) for e in cls)

    @classmethod
    def values(cls) -> tuple:
        return tuple(e.value for e in cls)


class StringEnumWithChoices(str, Enum):
    @classmethod
    def choices(cls) -> tuple[tuple[str, str], ...]:
        return tuple((str(e.value), str(e.value)) for e in cls)

    @classmethod
    def values(cls) -> tuple:
        return tuple(e.value for e in cls)
