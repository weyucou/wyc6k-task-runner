# Copyright (c) 2026 MNKT
"""Per-run workspace: a clone of the envelope's repository, with credentials and git identity configured.

The token reaches git and gh only as GH_TOKEN in the child environment; git asks the `gh auth git-credential`
helper for it. It is never in a URL, argv, `.git/config` or the prompt, and WorkspaceError messages have it
replaced by [REDACTED].
"""

import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from marvin.models import TaskEnvelope

DEFAULT_WORKSPACE_ROOT = "/workspace"  # WORKSPACE_ROOT overrides it
_CREDENTIAL_HELPER = "!gh auth git-credential"
_COMMAND_TIMEOUT_SECONDS = 600


class WorkspaceError(RuntimeError):
    """A clone step failed. The message never contains the token."""


@dataclass(frozen=True)
class Workspace:
    """The clone the agent runs in."""

    path: Path
    repository: str
    login: str  # the token's GitHub user; the clone commits as this user


class _GitHubUser(BaseModel):
    """The fields of `gh api user` the git identity is built from."""

    login: str
    id: int


def _scrub(message: str, token: str) -> str:
    # Guarded: str.replace("", ...) would put the marker between every character.
    return message.replace(token, "[REDACTED]") if token else message


def _run(argv: list[str], cwd: Path | None, token: str) -> str:
    """Run one git or gh command with the token as GH_TOKEN, and return its stdout.

    Raises WorkspaceError when the command cannot start, times out or exits non-zero. GIT_TERMINAL_PROMPT=0
    makes an auth failure fail at once instead of waiting on a prompt.
    """
    env = {**os.environ, "GH_TOKEN": token, "GIT_TERMINAL_PROMPT": "0"}
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkspaceError(_scrub(f"{shlex.join(argv)} failed: {exc}", token)) from exc
    if result.returncode != 0:
        raise WorkspaceError(_scrub(f"{shlex.join(argv)} failed: {result.stderr.strip()}", token))
    return result.stdout


def clone(envelope: TaskEnvelope, token: str, root: Path | None = None) -> Workspace:
    """Clone the envelope's repository into `<run dir>/<name>`, the run dir new under the workspace root.

    `root` defaults to WORKSPACE_ROOT, read at call time. The `gh auth git-credential` helper and the token
    user's identity are written to the clone's `.git/config` only, so the agent's git and gh commands need no
    setup. Raises WorkspaceError on any failure.
    """
    if not token:
        raise WorkspaceError(f"GitHub token required to clone {envelope.repository}")
    root = (root or Path(os.getenv("WORKSPACE_ROOT") or DEFAULT_WORKSPACE_ROOT)).resolve()

    # Before anything is written, so a bad token leaves nothing behind.
    try:
        user = _GitHubUser.model_validate_json(_run(["gh", "api", "user"], None, token))
    except ValidationError:
        raise WorkspaceError("gh api user returned no login and id") from None

    try:
        run_dir = Path(tempfile.mkdtemp(dir=root))
    except OSError as exc:
        raise WorkspaceError(f"cannot create a run directory under {root}: {exc}") from exc
    dest = (run_dir / envelope.repository.split("/", 1)[1]).resolve()
    # Backstop for the envelope's `.`/`..` check. `owner/..` resolves to the root itself, which
    # is_relative_to accepts.
    if dest == root or not dest.is_relative_to(root):
        raise WorkspaceError(f"clone destination {dest} is not inside the workspace root {root}")

    _run(
        [
            "git",
            "-c",
            "credential.helper=",
            "-c",
            f"credential.helper={_CREDENTIAL_HELPER}",
            "clone",
            "--filter=blob:none",
            f"https://github.com/{envelope.repository}.git",
            str(dest),
        ],
        None,
        token,
    )
    # The empty entry resets helpers inherited from the system or global config, as `-c` did for the clone.
    for helper in ("", _CREDENTIAL_HELPER):
        _run(["git", "config", "--local", "--add", "credential.helper", helper], dest, token)
    _run(["git", "config", "--local", "user.name", user.login], dest, token)
    _run(["git", "config", "--local", "user.email", f"{user.id}+{user.login}@users.noreply.github.com"], dest, token)
    return Workspace(path=dest, repository=envelope.repository, login=user.login)
