# Copyright (c) 2026 MNKT
"""Tests for entrypoint.py — one-shot Fargate runner."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import entrypoint
from marvin.context import CustomerContextBundle, MemoryEntry
from marvin.models import TaskEnvelope

PRE_CONTRACT_ENVELOPE_JSON = (Path(__file__).parent / "fixtures" / "envelopes" / "pre_contract_sender.json").read_text(
    encoding="utf-8"
)

LEAK_MARKER = "sk-ant-LEAKMARKER"
# agent.api_key is rejected at parse. It ends the JSON, where pydantic's truncated input_value would show it.
LEAKY_ENVELOPE_JSON = json.dumps(
    {
        "schema_version": 1,
        "task_id": "task-001",
        "customer_id": "cust-001",
        "task_source_id": 12,
        "repository": "acme-corp/billing-api",
        "issue_url": "https://github.com/acme-corp/billing-api/issues/42",
        "action": "develop",
        "context_s3_prefix": "s3://bucket/customers/cust-001/projects/billing-api/",
        "agent": {
            "provider": "anthropic",
            "model_name": "claude-sonnet-5-5",
            "tool_profile": "coding",
            "api_key": LEAK_MARKER,
        },
    }
)


def _make_envelope_dict(**kwargs) -> dict:
    defaults = {
        "schema_version": 1,
        "task_id": "task-001",
        "customer_id": "cust-001",
        "task_source_id": 12,
        "repository": "acme-corp/billing-api",
        "issue_url": "https://github.com/acme-corp/billing-api/issues/42",
        "action": "develop",
        "agent": {
            "provider": "anthropic",
            "model_name": "claude-sonnet-5-5",
            "tool_profile": "coding",
        },
        "context_s3_prefix": "s3://bucket/customers/cust-001/projects/billing-api/",
        "user_message": "Hello, agent!",
    }
    defaults.update(kwargs)
    return defaults


def _log_lines(capsys: pytest.CaptureFixture[str]) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def _make_context_bundle() -> CustomerContextBundle:
    return CustomerContextBundle(
        customer_id="cust-001",
        claude_md="# CLAUDE.md",
        sops={},
        project_goals="# Goals",
        memory_index="",
        daily_memories=[],
    )


class TestMainExitCodes:
    def test_missing_env_var_returns_2(self, monkeypatch) -> None:
        monkeypatch.delenv("TASK_ENVELOPE_JSON", raising=False)
        assert entrypoint.main() == 2

    def test_malformed_json_returns_2(self, monkeypatch) -> None:
        monkeypatch.setenv("TASK_ENVELOPE_JSON", "not-valid-json{{{")
        assert entrypoint.main() == 2

    def test_invalid_envelope_schema_returns_2(self, monkeypatch) -> None:
        monkeypatch.setenv("TASK_ENVELOPE_JSON", json.dumps({"task_id": "x"}))
        assert entrypoint.main() == 2

    @pytest.mark.parametrize(
        ("raw", "named"),
        [(LEAKY_ENVELOPE_JSON, "agent.api_key"), (LEAKY_ENVELOPE_JSON[:-1], "json_invalid")],
        ids=["api-key-rejected", "truncated-json"],
    )
    def test_invalid_envelope_logged_without_input(self, monkeypatch, capsys, raw, named) -> None:
        monkeypatch.setenv("TASK_ENVELOPE_JSON", raw)
        assert entrypoint.main() == 2
        out, err = capsys.readouterr()
        assert LEAK_MARKER not in out + err
        assert named in out

    def test_pre_contract_envelope_returns_2_naming_the_fields(self, monkeypatch, capsys) -> None:
        monkeypatch.setenv("TASK_ENVELOPE_JSON", PRE_CONTRACT_ENVELOPE_JSON)
        assert entrypoint.main() == 2
        [failure] = [line for line in _log_lines(capsys) if line["event"] == "Failed to parse task envelope"]
        locs = {tuple(error["loc"]) for error in failure["errors"]}
        assert locs == {
            ("schema_version",),
            ("task_id",),
            ("customer_id",),
            ("task_source_id",),
            ("repository",),
            ("agent",),
        }

    def test_credential_resolve_failure_returns_2(self, monkeypatch) -> None:
        monkeypatch.setenv("TASK_ENVELOPE_JSON", json.dumps(_make_envelope_dict()))
        with patch("entrypoint.CredentialResolver") as mock_resolver_cls:
            mock_resolver_cls.return_value.resolve.side_effect = RuntimeError("Secret fetch failed")
            assert entrypoint.main() == 2

    def test_context_pull_failure_returns_2(self, monkeypatch) -> None:
        monkeypatch.setenv("TASK_ENVELOPE_JSON", json.dumps(_make_envelope_dict()))
        with (
            patch("entrypoint.CredentialResolver.resolve", return_value=None),
            patch("entrypoint.ContextBundleService") as mock_svc_cls,
        ):
            mock_svc_cls.return_value.pull.side_effect = Exception("S3 unreachable")
            assert entrypoint.main() == 2

    def test_agent_run_failure_returns_1(self, monkeypatch) -> None:
        monkeypatch.setenv("TASK_ENVELOPE_JSON", json.dumps(_make_envelope_dict()))
        with (
            patch("entrypoint.CredentialResolver.resolve", return_value=None),
            patch("entrypoint.ContextBundleService") as mock_svc_cls,
            patch("entrypoint.AgentRunner") as mock_runner_cls,
        ):
            mock_svc_cls.return_value.pull.return_value = _make_context_bundle()
            mock_runner_cls.return_value.chat = AsyncMock(side_effect=RuntimeError("LLM timeout"))
            assert entrypoint.main() == 1

    def test_success_returns_0(self, monkeypatch) -> None:
        monkeypatch.setenv("TASK_ENVELOPE_JSON", json.dumps(_make_envelope_dict()))
        with (
            patch("entrypoint.CredentialResolver.resolve", return_value=None),
            patch("entrypoint.ContextBundleService") as mock_svc_cls,
            patch("entrypoint.AgentRunner") as mock_runner_cls,
        ):
            mock_svc = mock_svc_cls.return_value
            mock_svc.pull.return_value = _make_context_bundle()
            mock_svc.push_memory.return_value = None
            mock_runner_cls.return_value.chat = AsyncMock(return_value=("Agent reply", []))
            assert entrypoint.main() == 0

    def test_memory_write_failure_still_returns_0(self, monkeypatch) -> None:
        monkeypatch.setenv("TASK_ENVELOPE_JSON", json.dumps(_make_envelope_dict()))
        with (
            patch("entrypoint.CredentialResolver.resolve", return_value=None),
            patch("entrypoint.ContextBundleService") as mock_svc_cls,
            patch("entrypoint.AgentRunner") as mock_runner_cls,
        ):
            mock_svc = mock_svc_cls.return_value
            mock_svc.pull.return_value = _make_context_bundle()
            mock_svc.push_memory.side_effect = Exception("S3 write failed")
            mock_runner_cls.return_value.chat = AsyncMock(return_value=("Agent reply", []))
            assert entrypoint.main() == 0


class TestMainContextBundle:
    def test_context_bundle_passed_to_runner(self, monkeypatch) -> None:
        monkeypatch.setenv("TASK_ENVELOPE_JSON", json.dumps(_make_envelope_dict()))
        bundle = _make_context_bundle()
        with (
            patch("entrypoint.CredentialResolver.resolve", return_value=None),
            patch("entrypoint.ContextBundleService") as mock_svc_cls,
            patch("entrypoint.AgentRunner") as mock_runner_cls,
        ):
            mock_svc = mock_svc_cls.return_value
            mock_svc.pull.return_value = bundle
            mock_svc.push_memory.return_value = None
            mock_runner_cls.return_value.chat = AsyncMock(return_value=("reply", []))
            entrypoint.main()
            _, kwargs = mock_runner_cls.call_args
            assert kwargs["context_bundle"] is bundle

    def test_memory_entry_written_with_task_id(self, monkeypatch) -> None:
        envelope_dict = _make_envelope_dict(task_id="task-abc")
        monkeypatch.setenv("TASK_ENVELOPE_JSON", json.dumps(envelope_dict))
        with (
            patch("entrypoint.CredentialResolver.resolve", return_value=None),
            patch("entrypoint.ContextBundleService") as mock_svc_cls,
            patch("entrypoint.AgentRunner") as mock_runner_cls,
        ):
            mock_svc = mock_svc_cls.return_value
            mock_svc.pull.return_value = _make_context_bundle()
            mock_svc.push_memory.return_value = None
            mock_runner_cls.return_value.chat = AsyncMock(return_value=("summary text", []))
            entrypoint.main()
            assert mock_svc.push_memory.call_count == 1
            _, kwargs = mock_svc.push_memory.call_args
            entry: MemoryEntry = mock_svc.push_memory.call_args[0][1]
            assert "task-abc" in entry.content


class TestMainRunsTheTask:
    def _run(self, monkeypatch, envelope_dict: dict, api_key: str | None = None) -> tuple[MagicMock, MagicMock]:
        """Run main() to exit 0 with credentials, S3 and the runner patched.

        Returns the patched ContextBundleService instance and AgentRunner class.
        """
        monkeypatch.setenv("TASK_ENVELOPE_JSON", json.dumps(envelope_dict))
        with (
            patch("entrypoint.CredentialResolver.resolve", return_value=api_key),
            patch("entrypoint.ContextBundleService") as mock_svc_cls,
            patch("entrypoint.AgentRunner") as mock_runner_cls,
        ):
            mock_svc_cls.return_value.pull.return_value = _make_context_bundle()
            mock_runner_cls.return_value.chat = AsyncMock(return_value=("reply", []))
            assert entrypoint.main() == 0
        return mock_svc_cls.return_value, mock_runner_cls

    def test_runner_built_from_agent_block_and_resolved_key(self, monkeypatch) -> None:
        envelope_dict = _make_envelope_dict()
        _, mock_runner_cls = self._run(monkeypatch, envelope_dict, api_key="sk-resolved")
        kwargs = mock_runner_cls.call_args.kwargs
        assert kwargs["agent"] == TaskEnvelope.model_validate(envelope_dict).agent_config("sk-resolved")
        assert kwargs["session_id"] == "task-001"

    def test_chat_receives_the_task_prompt(self, monkeypatch) -> None:
        envelope_dict = _make_envelope_dict()
        _, mock_runner_cls = self._run(monkeypatch, envelope_dict)
        prompt = mock_runner_cls.return_value.chat.call_args.args[0]
        assert prompt == TaskEnvelope.model_validate(envelope_dict).task_prompt()

    def test_context_read_and_memory_written_at_context_s3_prefix(self, monkeypatch) -> None:
        mock_svc, _ = self._run(monkeypatch, _make_envelope_dict())
        prefix = "s3://bucket/customers/cust-001/projects/billing-api/"
        mock_svc.pull.assert_called_once_with(prefix)
        assert mock_svc.push_memory.call_args.args[0] == prefix

    def test_task_fields_bound_to_log_context(self, monkeypatch, capsys) -> None:
        self._run(monkeypatch, _make_envelope_dict())
        [complete] = [line for line in _log_lines(capsys) if line["event"] == "Task complete"]
        assert complete["repository"] == "acme-corp/billing-api"
        assert complete["issue_url"] == "https://github.com/acme-corp/billing-api/issues/42"
        assert complete["action"] == "develop"
        assert complete["task_source_id"] == 12


class TestBackgroundProcessCleanup:
    @pytest.mark.parametrize(
        ("chat", "exit_code"),
        [({"return_value": ("reply", [])}, 0), ({"side_effect": RuntimeError("LLM timeout")}, 1)],
        ids=["run-succeeds", "run-raises"],
    )
    def test_background_processes_stopped_when_the_run_ends(self, monkeypatch, chat, exit_code) -> None:
        monkeypatch.setenv("TASK_ENVELOPE_JSON", json.dumps(_make_envelope_dict()))
        with (
            patch("entrypoint.CredentialResolver.resolve", return_value=None),
            patch("entrypoint.ContextBundleService") as mock_svc_cls,
            patch("entrypoint.AgentRunner") as mock_runner_cls,
            patch("entrypoint.stop_background_processes") as mock_stop,
        ):
            mock_svc_cls.return_value.pull.return_value = _make_context_bundle()
            mock_runner_cls.return_value.chat = AsyncMock(**chat)
            assert entrypoint.main() == exit_code
        mock_stop.assert_called_once_with()


class TestScrubSensitiveFields:
    def test_scrubs_known_sensitive_keys(self) -> None:
        event_dict = {"event": "test", "token": "secret-value", "password": "hunter2"}
        result = entrypoint._scrub_sensitive_fields(None, "info", event_dict)
        assert result["token"] == "[REDACTED]"
        assert result["password"] == "[REDACTED]"

    def test_preserves_non_sensitive_keys(self) -> None:
        event_dict = {"event": "test", "step": "parse", "status": "ok"}
        result = entrypoint._scrub_sensitive_fields(None, "info", event_dict)
        assert result["step"] == "parse"
        assert result["status"] == "ok"
