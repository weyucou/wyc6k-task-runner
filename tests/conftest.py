# Copyright (c) 2026 MNKT
"""Fixtures shared by the unit and integration suites."""

import subprocess
from pathlib import Path

import pytest

_SEED_IDENTITY = ("-c", "user.name=Fake Remote", "-c", "user.email=fake-remote@example.com")


def _git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


class FakeRemote:
    """Local bare repositories that stand in for github.com.

    `fake_remote` routes `https://github.com/` to `file://<remotes>/` with a `url.<base>.insteadOf` rule in
    GIT_CONFIG_* env vars, so the production clone URL is used unchanged and no network is needed. Git never
    calls a credential helper for file://, so `gh auth git-credential` is not exercised here.
    """

    def __init__(self, base: Path) -> None:
        self.remotes = base / "remotes"
        self.workspace_root = base / "workspace"
        self.global_config = base / "gitconfig-global"
        self._seeds = base / "seeds"

    def add(self, repository: str) -> Path:
        """Serve `repository` (`owner/name`) with one commit on `main`; return the bare repository."""
        seed = self._seeds / repository
        seed.mkdir(parents=True)
        _git("init", "--initial-branch=main", cwd=seed)
        (seed / "README.md").write_text(f"# {repository}\n", encoding="utf-8")
        _git("add", "README.md", cwd=seed)
        _git(*_SEED_IDENTITY, "commit", "--message=Initial commit", cwd=seed)
        bare = self.remotes / f"{repository}.git"
        _git("clone", "--bare", str(seed), str(bare))
        return bare


@pytest.fixture
def fake_remote(tmp_path, monkeypatch) -> FakeRemote:
    """Serve github.com from local bare repositories, with WORKSPACE_ROOT under tmp_path.

    Git reads an empty global config and no system config, so the developer's own gitconfig cannot change a
    test, and a test can check that nothing was written globally. The cwd is restored at teardown, because
    `entrypoint.main()` moves into the clone.
    """
    remote = FakeRemote(tmp_path)
    remote.remotes.mkdir()
    remote.workspace_root.mkdir()
    remote.global_config.touch()
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(remote.global_config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", f"url.{remote.remotes.as_uri()}/.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "https://github.com/")
    monkeypatch.setenv("WORKSPACE_ROOT", str(remote.workspace_root))
    monkeypatch.chdir(Path.cwd())
    return remote
