import base64
import logging

import httpx

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"


def _github_headers(token: str = "") -> dict:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_readme(repo_owner: str, repo_name: str, token: str = "") -> str:
    """Fetch README content from GitHub. Returns empty string if not found."""
    url = f"{GITHUB_API_BASE}/repos/{repo_owner}/{repo_name}/readme"
    try:
        response = httpx.get(url, headers=_github_headers(token), timeout=10)
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
