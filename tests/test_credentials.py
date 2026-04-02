"""Tests for marvin.credentials — CredentialResolver."""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from marvin.credentials import CredentialResolver, _fetch_secret
from marvin.models import AgentConfig, TaskEnvelope


def _make_envelope(**kwargs) -> TaskEnvelope:
    defaults = dict(
        task_id="t-001",
        customer_id="c-001",
        session_id="s-001",
        agent=AgentConfig(name="test-agent"),
        s3_context_prefix="s3://bucket/prefix",
        user_message="Hello",
    )
    defaults.update(kwargs)
    return TaskEnvelope(**defaults)


class TestFetchSecret:
    def test_returns_plain_string(self) -> None:
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": "my-plain-token"}
        with patch("marvin.credentials.get_secrets_manager_client", return_value=mock_client):
            result = _fetch_secret("arn:aws:secretsmanager:ap-northeast-1:123:secret:tok")
        assert result == "my-plain-token"

    def test_returns_value_from_json_object(self) -> None:
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": '{"value": "json-token"}'}
        with patch("marvin.credentials.get_secrets_manager_client", return_value=mock_client):
            result = _fetch_secret("arn:aws:secretsmanager:ap-northeast-1:123:secret:tok")
        assert result == "json-token"

    def test_raises_runtime_error_on_fetch_failure(self) -> None:
        mock_client = MagicMock()
        mock_client.get_secret_value.side_effect = Exception("AccessDenied")
        with patch("marvin.credentials.get_secrets_manager_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Failed to fetch secret"):
                _fetch_secret("arn:aws:secretsmanager:ap-northeast-1:123:secret:tok")

    def test_json_without_value_key_returns_raw(self) -> None:
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": '{"token": "other-format"}'}
        with patch("marvin.credentials.get_secrets_manager_client", return_value=mock_client):
            result = _fetch_secret("arn:aws:secretsmanager:ap-northeast-1:123:secret:tok")
        assert result == '{"token": "other-format"}'


class TestCredentialResolver:
    def test_github_token_secret_id_sets_env_var(self, monkeypatch) -> None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        envelope = _make_envelope(github_token_secret_id="arn:aws:secretsmanager:ap-northeast-1:123:secret:gh")
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": "gh-tok-abc"}
        with patch("marvin.credentials.get_secrets_manager_client", return_value=mock_client):
            CredentialResolver().resolve(envelope)
        assert os.environ["GITHUB_TOKEN"] == "gh-tok-abc"

    def test_anthropic_key_secret_id_sets_agent_api_key(self) -> None:
        envelope = _make_envelope(anthropic_api_key_secret_id="arn:aws:secretsmanager:ap-northeast-1:123:secret:ak")
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": "sk-ant-123"}
        with patch("marvin.credentials.get_secrets_manager_client", return_value=mock_client):
            CredentialResolver().resolve(envelope)
        assert envelope.agent.api_key == "sk-ant-123"

    def test_fallback_to_env_github_token(self, monkeypatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "env-gh-token")
        envelope = _make_envelope()  # no secret ID fields
        CredentialResolver().resolve(envelope)
        assert os.environ["GITHUB_TOKEN"] == "env-gh-token"

    def test_fallback_to_env_anthropic_key(self, monkeypatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-ant-key")
        envelope = _make_envelope()  # no secret ID fields
        CredentialResolver().resolve(envelope)
        assert envelope.agent.api_key == "env-ant-key"

    def test_raises_before_processing_on_secret_fetch_failure(self) -> None:
        envelope = _make_envelope(github_token_secret_id="arn:aws:secretsmanager:ap-northeast-1:123:secret:bad")
        mock_client = MagicMock()
        mock_client.get_secret_value.side_effect = Exception("ResourceNotFoundException")
        with patch("marvin.credentials.get_secrets_manager_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Failed to fetch secret"):
                CredentialResolver().resolve(envelope)

    def test_no_secret_ids_no_env_vars_resolves_none(self, monkeypatch) -> None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        envelope = _make_envelope()
        CredentialResolver().resolve(envelope)
        # No env var set, no api_key set — should not raise
        assert envelope.agent.api_key is None

    def test_does_not_log_secret_value(self, monkeypatch, caplog) -> None:
        """CredentialResolver must never log resolved secret values."""
        envelope = _make_envelope(
            github_token_secret_id="arn:aws:secretsmanager:ap-northeast-1:123:secret:gh",
            anthropic_api_key_secret_id="arn:aws:secretsmanager:ap-northeast-1:123:secret:ak",
        )
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": "super-secret-value-xyz"}
        with patch("marvin.credentials.get_secrets_manager_client", return_value=mock_client):
            with caplog.at_level(logging.DEBUG, logger="marvin.credentials"):
                CredentialResolver().resolve(envelope)
        for record in caplog.records:
            assert "super-secret-value-xyz" not in record.message
