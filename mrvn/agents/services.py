import logging
from urllib.parse import urlparse

import boto3
from django.utils import timezone

logger = logging.getLogger(__name__)


def _list_s3_sop_files(s3_prefix: str) -> dict[str, str]:
    """List and read SOP files under s3_prefix. Returns {key: content}."""
    parsed = urlparse(s3_prefix)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    s3 = boto3.client("s3")
    result: dict[str, str] = {}
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                try:
                    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                    result[key] = body.decode("utf-8")
                except Exception as exc:
                    logger.warning("Failed to read S3 object %s: %s", key, exc)
    except Exception as exc:
        logger.warning("Failed to list S3 prefix %s: %s", s3_prefix, exc)
    return result


def sync_project_context(project_context_id: int) -> None:
    """Sync sops_snapshot for a ProjectContext from S3.

    Reads SOP files from the configured S3 prefix and updates
    sops_snapshot and last_synced.
    """
    from agents.models import ProjectContext

    ctx = ProjectContext.objects.get(pk=project_context_id)

    sops = _list_s3_sop_files(ctx.s3_prefix)

    ctx.sops_snapshot = sops
    ctx.last_synced = timezone.now()
    ctx.save(update_fields=["sops_snapshot", "last_synced", "updated_datetime"])

    logger.info(
        "Synced ProjectContext pk=%s (project=%s): %d SOP files",
        ctx.pk,
        ctx.project_id,
        len(sops),
    )
