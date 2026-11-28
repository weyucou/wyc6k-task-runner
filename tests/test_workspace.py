# Copyright (c) 2026 MNKT
"""Tests for marvin.workspace — the per-run clone of the envelope's repository.

No network: `fake_remote` (tests/conftest.py) serves github.com from local bare repositories, and a `gh` stub on
PATH answers `gh api user`.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import entrypoint
from marvin import workspace as workspace_module
from marvin.models import TaskEnvelope
from marvin.workspace import Workspace, WorkspaceError, clone
from tests.conftest import FakeRemote

REPOSITORY = "acme-corp/billing-api"
TOKEN = "workspace-github-token-not-a-real-credential"
LOGIN = "sandbox-bot"
USER_ID = 1234

# `gh` stand-in: records each call with the GH_TOKEN it ran with, and answers `api user`.
# With GH_STUB_FAIL set, `api user` fails and echoes the token to stderr, as a careless CLI might.
_GH_STUB_SOURCE = '''#!{python}
"""Recording stand-in for the `gh` CLI."""

import json
import os
import pathlib
import sys

argv = sys.argv[1:]
record = json.dumps(dict(argv=argv, gh_token=os.environ.get("GH_TOKEN", "")))
with pathlib.Path({record_path!r}).open("a", encoding="utf-8") as handle:
    handle.write(record + "\\n")

if argv == ["api", "user"]:
    if os.environ.get("GH_STUB_FAIL"):
        sys.stderr.write("HTTP 401: Bad credentials: " + os.environ.get("GH_TOKEN", "") + "\\n")
        sys.exit(1)
    sys.stdout.write(json.dumps(dict(login={login!r}, id={user_id})) + "\\n")
'''


class GhStub:
    """Reads back the calls recorded by the `gh` stub."""

    def __init__(self, record_path: Path) -> None:
        self._record_path = record_path

    @property
    def calls(self) -> list[dict[str, Any]]:
        if not self._record_path.exists():
            return []
        return [json.loads(line) for line in self._record_path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(autouse=True)
def gh(tmp_path, monkeypatch) -> GhStub:
    """Put the recording `gh` stand-in at the front of PATH, so no test here can reach the real `gh`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record_path = tmp_path / "gh-calls.jsonl"
    stub = bin_dir / "gh"
    stub.write_text(
        _GH_STUB_SOURCE.format(python=sys.executable, record_path=str(record_path), login=LOGIN, user_id=USER_ID),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("GH_STUB_FAIL", raising=False)
    return GhStub(record_path)


@pytest.fixture
def remote(fake_remote) -> FakeRemote:
    """The fake remote, serving REPOSITORY."""
    fake_remote.add(REPOSITORY)
    return fake_remote


def _envelope_dict(repository: str = REPOSITORY, **kwargs: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": "task-001",
        "customer_id": "cust-001",
        "task_source_id": 12,
        "repository": repository,
        "issue_url": f"https://github.com/{repository}/issues/42",
        "action": "develop",
        "agent": {"provider": "anthropic", "model_name": "claude-sonnet-5-5", "tool_profile": "coding"},
        "context_s3_prefix": "s3://bucket/customers/cust-001/projects/billing-api/",
        **kwargs,
    }


def _envelope(repository: str = REPOSITORY, **kwargs: Any) -> TaskEnvelope:
    return TaskEnvelope.model_validate(_envelope_dict(repository, **kwargs))


def _local_config(repo: Path, *args: str) -> list[str]:
    """Values from the clone's own `.git/config`."""
    result = subprocess.run(["git", "config", "--local", *args], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.splitlines()


class TestClone:
    def test_clones_into_its_own_directory_under_the_root(self, remote) -> None:
        workspace = clone(_envelope(), TOKEN)
        assert workspace == Workspace(path=workspace.path, repository=REPOSITORY, login=LOGIN)
        assert workspace.path.name == "billing-api"
        assert workspace.path.parent.parent == remote.workspace_root.resolve()
        assert (workspace.path / ".git").is_dir()
        assert (workspace.path / "README.md").read_text(encoding="utf-8") == f"# {REPOSITORY}\n"

    def test_git_identity_and_credential_helper_set_in_the_clone(self, remote) -> None:
        workspace = clone(_envelope(), TOKEN)
        assert _local_config(workspace.path, "user.name") == [LOGIN]
        assert _local_config(workspace.path, "user.email") == [f"{USER_ID}+{LOGIN}@users.noreply.github.com"]
        # The empty entry resets any helper inherited from the system or global config.
        assert _local_config(workspace.path, "--get-all", "credential.helper") == ["", "!gh auth git-credential"]

    def test_token_kept_out_of_argv_url_and_git_config(self, remote, gh) -> None:
        with patch("marvin.workspace.subprocess.run", wraps=subprocess.run) as run:
            workspace = clone(_envelope(), TOKEN)
        assert run.call_count > 0
        for call in run.call_args_list:
            assert all(TOKEN not in arg for arg in call.args[0])
            assert call.kwargs["env"]["GH_TOKEN"] == TOKEN
        assert gh.calls == [{"argv": ["api", "user"], "gh_token": TOKEN}]
        assert _local_config(workspace.path, "remote.origin.url") == [f"https://github.com/{REPOSITORY}.git"]
        assert TOKEN not in (workspace.path / ".git" / "config").read_text(encoding="utf-8")
        assert remote.global_config.read_text(encoding="utf-8") == ""

    def test_task_id_not_used_in_the_path(self, remote) -> None:
        workspace = clone(_envelope(task_id="../escape"), TOKEN)
        assert workspace.path.parent.parent == remote.workspace_root.resolve()
        assert "escape" not in str(workspace.path)


class TestCloneFailures:
    def test_missing_remote(self, remote) -> None:
        with pytest.raises(WorkspaceError, match=r"git .* clone .* failed: ") as exc_info:
            clone(_envelope("acme-corp/does-not-exist"), TOKEN)
        assert TOKEN not in str(exc_info.value)

    def test_gh_api_user_failure_scrubs_the_token(self, remote, monkeypatch) -> None:
        monkeypatch.setenv("GH_STUB_FAIL", "1")
        with pytest.raises(WorkspaceError, match="gh api user failed") as exc_info:
            clone(_envelope(), TOKEN)
        assert TOKEN not in str(exc_info.value)
        assert "Bad credentials: [REDACTED]" in str(exc_info.value)
        assert list(remote.workspace_root.iterdir()) == []  # it fails before anything is written

    def test_gh_not_installed(self, fake_remote, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("PATH", str(tmp_path / "no-bin"))
        with pytest.raises(WorkspaceError, match="gh api user failed"):
            clone(_envelope(), TOKEN)

    def test_missing_workspace_root(self, remote, tmp_path) -> None:
        with pytest.raises(WorkspaceError, match="cannot create a run directory"):
            clone(_envelope(), TOKEN, root=tmp_path / "missing")

    def test_empty_token_fails_before_gh_runs(self, remote, gh) -> None:
        with pytest.raises(WorkspaceError, match="GitHub token required"):
            clone(_envelope(), "")
        assert gh.calls == []
        assert list(remote.workspace_root.iterdir()) == []

    @pytest.mark.parametrize("repository", ["acme-corp/..", "acme-corp/../../escape"], ids=["root", "outside-root"])
    def test_destination_outside_the_root_rejected_before_cloning(self, remote, repository) -> None:
        envelope = TaskEnvelope.model_construct(repository=repository)  # skips the envelope's own check
        with (
            patch("marvin.workspace.subprocess.run", wraps=subprocess.run) as run,
            pytest.raises(WorkspaceError, match="not inside the workspace root"),
        ):
            clone(envelope, TOKEN)
        assert [call.args[0] for call in run.call_args_list] == [["gh", "api", "user"]]


class TestRun:
    def test_failure_message_kept_when_there_is_no_token(self) -> None:
        # Scrubbing an empty token would put the marker between every character.
        with pytest.raises(WorkspaceError) as exc_info:
            workspace_module._run([sys.executable, "-c", "import sys; sys.exit('boom')"], None, "")
        assert str(exc_info.value).endswith(" failed: boom")


def _log_lines(out: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in out.splitlines()]


def _run_main(monkeypatch, envelope_dict: dict[str, Any]) -> tuple[int, MagicMock, list[Path]]:
    """Run entrypoint.main() through the real clone step, with credentials, S3 and the runner patched.

    Returns the exit code, the patched AgentRunner class and the cwd the agent ran in.
    """
    monkeypatch.setenv("TASK_ENVELOPE_JSON", json.dumps(envelope_dict))
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    agent_cwd: list[Path] = []

    def chat(*_args: Any, **_kwargs: Any) -> tuple[str, list[Any]]:
        agent_cwd.append(Path.cwd())
        return "reply", []

    with (
        patch("entrypoint.CredentialResolver.resolve", return_value=None),
        patch("entrypoint.ContextBundleService"),
        patch("entrypoint.AgentRunner") as mock_runner_cls,
    ):
        mock_runner_cls.return_value.chat = AsyncMock(side_effect=chat)
        exit_code = entrypoint.main()
    return exit_code, mock_runner_cls, agent_cwd


class TestEntrypointCloneStep:
    def test_agent_runs_in_the_clone(self, monkeypatch, capsys, remote) -> None:
        exit_code, mock_runner_cls, agent_cwd = _run_main(monkeypatch, _envelope_dict())
        assert exit_code == 0
        out, err = capsys.readouterr()
        [cloned] = [line for line in _log_lines(out) if line.get("step") == "clone" and line["status"] == "ok"]
        path = Path(cloned["workspace"])
        assert agent_cwd == [path]
        assert path.name == "billing-api"
        assert path.parent.parent == remote.workspace_root.resolve()
        assert TOKEN not in out + err + str(mock_runner_cls.return_value.chat.call_args)

    @pytest.mark.parametrize(
        ("repository", "gh_stub_fail"),
        [("acme-corp/does-not-exist", ""), (REPOSITORY, "1")],
        ids=["missing-remote", "gh-api-user-fails"],
    )
    def test_clone_failure_returns_2_without_the_token(
        self, monkeypatch, capsys, remote, repository, gh_stub_fail
    ) -> None:
        monkeypatch.setenv("GH_STUB_FAIL", gh_stub_fail)
        exit_code, mock_runner_cls, _ = _run_main(monkeypatch, _envelope_dict(repository))
        assert exit_code == 2
        mock_runner_cls.assert_not_called()
        out, err = capsys.readouterr()
        assert TOKEN not in out + err
        [failed] = [line for line in _log_lines(out) if line["event"] == "Failed to clone repository"]
        assert failed["step"] == "clone"
        assert failed["status"] == "failed"
