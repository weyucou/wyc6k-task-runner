import base64
import json
import logging
import os
from urllib.parse import urlparse

import boto3
import httpx
from django.utils import timezone

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"


def _github_headers() -> dict:
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_readme(repo_owner: str, repo_name: str) -> str:
    """Fetch README content from GitHub. Returns empty string if not found."""
    url = f"{GITHUB_API_BASE}/repos/{repo_owner}/{repo_name}/readme"
    try:
        response = httpx.get(url, headers=_github_headers(), timeout=10)
        if response.status_code == 404:
            logger.info("No README found for %s/%s", repo_owner, repo_name)
            return ""
        response.raise_for_status()
        data = response.json()
        encoded = data.get("content", "")
        return base64.b64decode(encoded).decode("utf-8")
    except Exception as exc:
        logger.warning("Failed to fetch README for %s/%s: %s", repo_owner, repo_name, exc)
        return ""


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
    """Sync goals_markdown and sops_snapshot for a ProjectContext from GitHub and S3.

    Reads the repository README from GitHub and SOP files from the configured
    S3 prefix, then updates goals_markdown, sops_snapshot, and last_synced.
    """
    from agents.models import ProjectContext

    ctx = ProjectContext.objects.get(pk=project_context_id)

    goals = _fetch_readme(ctx.repo_owner, ctx.repo_name)
    sops = _list_s3_sop_files(ctx.s3_prefix)

    ctx.goals_markdown = goals
    ctx.sops_snapshot = sops
    ctx.last_synced = timezone.now()
    ctx.save(update_fields=["goals_markdown", "sops_snapshot", "last_synced", "updated_datetime"])

    logger.info(
        "Synced ProjectContext pk=%s (%s/%s): %d chars README, %d SOP files",
        ctx.pk,
        ctx.repo_owner,
        ctx.repo_name,
        len(goals),
        len(sops),
    )
