# Copyright (c) 2026 MNKT
"""Tests for marvin.models — AgentConfig, TaskEnvelope (v1 contract), ToolProfile."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from marvin.models import AgentConfig, LLMProvider, TaskEnvelope, ToolProfile

# Spec examples (weyucou/wyc6k-spec docs/architecture.md §TaskEnvelope), committed verbatim.
ENVELOPE_FIXTURES = Path(__file__).parent / "fixtures" / "envelopes"
V1_EXAMPLES = ["v1_develop_bedrock.json", "v1_review_anthropic.json"]

REPOSITORY = "acme-corp/billing-api"
ISSUE_URL = f"https://github.com/{REPOSITORY}/issues/42"
AGENT = {"provider": "anthropic", "model_name": "claude-opus-5-5", "tool_profile": "coding"}
LEAK_MARKER = "sk-ant-LEAKMARKER"


def _fixture(name: str) -> str:
    return (ENVELOPE_FIXTURES / name).read_text(encoding="utf-8")


def _envelope_dict(**kwargs: Any) -> dict[str, Any]:
    """A valid v1 envelope with only the required fields; kwargs replace top-level fields."""
    envelope = {
        "schema_version": 1,
        "task_id": "t-001",
        "customer_id": "7",
        "task_source_id": 12,
        "repository": REPOSITORY,
        "issue_url": ISSUE_URL,
        "action": "develop",
        "agent": AGENT,
        "context_s3_prefix": "s3://bucket/acme-corp/projects/billing-api/",
    }
    envelope.update(kwargs)
    return envelope


def _without(field: str) -> dict[str, Any]:
    return {key: value for key, value in _envelope_dict().items() if key != field}


def _error_locs(exc: ValidationError) -> set[tuple[int | str, ...]]:
    return {error["loc"] for error in exc.errors()}


# agent.api_key is rejected at parse. It ends the input, where pydantic's truncated input_value would show it.
LEAKY_ENVELOPE = _without("agent") | {"agent": {**AGENT, "api_key": LEAK_MARKER}}


class TestAgentConfig:
    def test_defaults(self) -> None:
        agent = AgentConfig(name="test-agent")
        assert agent.provider == LLMProvider.ANTHROPIC
        assert agent.model_name == "claude-sonnet-5-5"
        assert agent.temperature == 0.7
        assert agent.max_tokens == 4096
        assert agent.is_active is True
        assert agent.rate_limit_enabled is False
        assert agent.tool_profile == ToolProfile.FULL

    def test_name_required(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig()  # type: ignore[call-arg]

    def test_all_providers_valid(self) -> None:
        for provider in LLMProvider:
            agent = AgentConfig(name="agent", provider=provider)
            assert agent.provider == provider

    def test_api_key_optional(self) -> None:
        agent = AgentConfig(name="agent", api_key="sk-test-key")
        assert agent.api_key == "sk-test-key"

        agent_no_key = AgentConfig(name="agent")
        assert agent_no_key.api_key is None


class TestAgentConfigGetAllowedTools:
    def _make_available(self) -> list[str]:
        return ["read", "write", "exec", "get_datetime", "calculator", "send_email", "memory_store"]

    def test_full_profile_returns_all(self) -> None:
        agent = AgentConfig(name="a", tool_profile=ToolProfile.FULL)
        available = self._make_available()
        allowed = agent.get_allowed_tools(available)
        assert set(allowed) == set(available)

    def test_minimal_profile_returns_none(self) -> None:
        agent = AgentConfig(name="a", tool_profile=ToolProfile.MINIMAL)
        allowed = agent.get_allowed_tools(self._make_available())
        assert allowed == []

    def test_coding_profile_filters(self) -> None:
        agent = AgentConfig(name="a", tool_profile=ToolProfile.CODING)
        available = ["read", "write", "exec", "send_email", "get_datetime"]
        allowed = agent.get_allowed_tools(available)
        assert "read" in allowed
        assert "write" in allowed
        assert "exec" in allowed
        assert "send_email" not in allowed

    def test_messaging_profile_filters(self) -> None:
        agent = AgentConfig(name="a", tool_profile=ToolProfile.MESSAGING)
        available = ["send_email", "slack_message", "read", "exec"]
        allowed = agent.get_allowed_tools(available)
        assert "send_email" in allowed
        assert "slack_message" in allowed
        assert "read" not in allowed
        assert "exec" not in allowed

    def test_tools_deny_removes_from_allowed(self) -> None:
        agent = AgentConfig(name="a", tool_profile=ToolProfile.FULL, tools_deny=["exec", "write"])
        available = ["read", "write", "exec", "get_datetime"]
        allowed = agent.get_allowed_tools(available)
        assert "exec" not in allowed
        assert "write" not in allowed
        assert "read" in allowed

    def test_tools_allow_adds_to_profile(self) -> None:
        agent = AgentConfig(name="a", tool_profile=ToolProfile.MINIMAL, tools_allow=["get_datetime"])
        available = ["read", "get_datetime", "exec"]
        allowed = agent.get_allowed_tools(available)
        assert "get_datetime" in allowed
        assert "read" not in allowed

    def test_empty_available_returns_empty(self) -> None:
        agent = AgentConfig(name="a", tool_profile=ToolProfile.FULL)
        assert agent.get_allowed_tools([]) == []


class TestSpecExamples:
    @pytest.mark.parametrize("name", V1_EXAMPLES)
    def test_validates(self, name) -> None:
        envelope = TaskEnvelope.model_validate_json(_fixture(name))
        assert envelope.schema_version == 1
        assert envelope.session_id == envelope.task_id
        assert envelope.conversation_history == []
        assert envelope.enable_tools is True

    def test_develop_on_bedrock(self) -> None:
        envelope = TaskEnvelope.model_validate_json(_fixture("v1_develop_bedrock.json"))
        assert envelope.action == "develop"
        assert envelope.agent.provider == "bedrock"
        assert envelope.agent.model_name == "anthropic.claude-opus-5-5"
        assert envelope.agent.tool_profile == ToolProfile.CODING
        assert envelope.agent.max_tokens == 16000
        assert envelope.anthropic_api_key_secret_id is None
        assert envelope.user_message == ""

    def test_review_on_anthropic_applies_defaults(self) -> None:
        envelope = TaskEnvelope.model_validate_json(_fixture("v1_review_anthropic.json"))
        assert envelope.action == "review"
        assert envelope.agent.provider == LLMProvider.ANTHROPIC
        assert envelope.agent.tool_profile == ToolProfile.FULL
        assert envelope.agent.max_tokens == 4096
        assert envelope.agent.tools_allow == []
        assert envelope.agent.tools_deny == []
        assert envelope.max_tool_iterations == 10
        assert envelope.anthropic_api_key_secret_id == "weyucou/agent/acme-corp/anthropic"
        assert envelope.user_message == "Focus the review on error handling and test coverage."

    def test_pre_contract_sender_shape_fails_naming_the_fields(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            TaskEnvelope.model_validate_json(_fixture("pre_contract_sender.json"))
        assert _error_locs(exc_info.value) == {
            ("schema_version",),
            ("task_id",),
            ("customer_id",),  # an int; the contract's customer_id is a str
            ("task_source_id",),
            ("repository",),
            ("agent",),  # the top-level agent_profile is an unknown field, so it is ignored
        }


class TestTaskEnvelopeValidation:
    @pytest.mark.parametrize(
        ("data", "loc"),
        [
            pytest.param(_envelope_dict(schema_version=2), ("schema_version",), id="unknown-schema-version"),
            pytest.param(_without("schema_version"), ("schema_version",), id="missing-schema-version"),
            pytest.param(_without("repository"), ("repository",), id="missing-repository"),
            pytest.param(_envelope_dict(repository="billing-api"), ("repository",), id="repository-not-owner-name"),
            pytest.param(
                _envelope_dict(issue_url="https://github.com/acme-corp/payments-api/issues/42"),
                ("issue_url",),
                id="issue-url-in-another-repository",
            ),
            pytest.param(
                _envelope_dict(issue_url="https://github.com/acme-corp/billing-api-v2/issues/42"),
                ("issue_url",),
                id="issue-url-in-repository-with-same-prefix",
            ),
            pytest.param(
                _envelope_dict(issue_url="https://github.com/acme-corp/billing-api"),
                ("issue_url",),
                id="issue-url-not-an-issue",
            ),
            pytest.param(
                _envelope_dict(issue_url="https://github.example.com/acme-corp/billing-api/issues/42"),
                ("issue_url",),
                id="issue-url-on-another-host",
            ),
            pytest.param(_envelope_dict(action="prepare"), ("action",), id="unknown-action"),
            pytest.param(_envelope_dict(customer_id=7), ("customer_id",), id="int-customer-id"),
            pytest.param(
                _envelope_dict(agent={**AGENT, "tool_profile": "CODING"}),
                ("agent", "tool_profile"),
                id="uppercase-tool-profile",
            ),
            pytest.param(
                _envelope_dict(agent={"model_name": "claude-opus-5-5", "tool_profile": "coding"}),
                ("agent", "provider"),
                id="missing-provider",
            ),
            pytest.param(_envelope_dict(agent={**AGENT, "api_key": "sk-ant-x"}), ("agent",), id="agent-api-key"),
        ],
    )
    def test_invalid_field_named_in_error(self, data, loc) -> None:
        with pytest.raises(ValidationError) as exc_info:
            TaskEnvelope.model_validate_json(json.dumps(data))
        assert _error_locs(exc_info.value) == {loc}

    @pytest.mark.parametrize(
        "issue_url",
        [
            ISSUE_URL,
            f"https://github.com/{REPOSITORY}/pull/57",
            "https://github.com/Acme-Corp/Billing-API/issues/42",
        ],
        ids=["issue", "pull-request", "different-case"],
    )
    def test_issue_url_in_repository_accepted(self, issue_url) -> None:
        assert TaskEnvelope.model_validate(_envelope_dict(issue_url=issue_url)).issue_url == issue_url

    def test_agent_api_key_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            TaskEnvelope.model_validate_json(json.dumps(LEAKY_ENVELOPE))
        [error] = exc_info.value.errors()
        assert error["loc"] == ("agent",)
        assert "agent.api_key is not accepted" in error["msg"]

    @pytest.mark.parametrize(
        "raw",
        [
            json.dumps(LEAKY_ENVELOPE),
            json.dumps({key: value for key, value in LEAKY_ENVELOPE.items() if key != "context_s3_prefix"}),
            json.dumps(LEAKY_ENVELOPE)[:-1],
        ],
        ids=["api-key-rejected", "missing-field", "truncated-json"],
    )
    def test_validation_error_hides_input(self, raw) -> None:
        with pytest.raises(ValidationError) as exc_info:
            TaskEnvelope.model_validate_json(raw)
        assert LEAK_MARKER not in str(exc_info.value)

    def test_optional_fields_default(self) -> None:
        envelope = TaskEnvelope.model_validate(_envelope_dict())
        assert envelope.duration_hint_seconds == 3600
        assert envelope.max_tool_iterations == 10
        assert envelope.github_token_secret_id is None
        assert envelope.anthropic_api_key_secret_id is None
        assert envelope.user_message == ""
        assert envelope.agent.max_tokens == 4096
        assert envelope.agent.tools_allow == []
        assert envelope.agent.tools_deny == []

    def test_session_id_derived_from_task_id(self) -> None:
        assert TaskEnvelope.model_validate(_envelope_dict()).session_id == "t-001"
        assert TaskEnvelope.model_validate(_envelope_dict(session_id="s-001")).session_id == "s-001"

    def test_runner_options_accepted(self) -> None:
        history = [{"role": "user", "content": "Previous message"}]
        envelope = TaskEnvelope.model_validate(_envelope_dict(conversation_history=history, enable_tools=False))
        assert envelope.conversation_history == history
        assert envelope.enable_tools is False

    def test_unknown_fields_ignored(self) -> None:
        # Runner-internal AgentConfig fields are not part of the wire contract: a sender cannot set them.
        agent = {**AGENT, "name": "sender-agent", "base_url": "https://llm.example.com", "system_prompt": "Hi"}
        envelope = TaskEnvelope.model_validate(_envelope_dict(agent=agent, agent_profile="CODING"))
        config = envelope.agent_config(None)
        assert config.name == f"{REPOSITORY}#develop"
        assert config.base_url == ""
        assert config.system_prompt == ""


class TestTaskEnvelopeAgentConfig:
    def test_built_from_agent_block_and_resolved_key(self) -> None:
        agent = {**AGENT, "max_tokens": 16000, "tools_allow": ["read"], "tools_deny": ["exec"]}
        envelope = TaskEnvelope.model_validate(_envelope_dict(action="review", agent=agent))
        config = envelope.agent_config("sk-resolved")
        assert isinstance(config, AgentConfig)
        assert config.name == f"{REPOSITORY}#review"
        assert config.provider == LLMProvider.ANTHROPIC
        assert config.model_name == "claude-opus-5-5"
        assert config.tool_profile == ToolProfile.CODING
        assert config.max_tokens == 16000
        assert config.tools_allow == ["read"]
        assert config.tools_deny == ["exec"]
        assert config.api_key == "sk-resolved"

    def test_no_resolved_key(self) -> None:
        assert TaskEnvelope.model_validate(_envelope_dict()).agent_config(None).api_key is None


class TestTaskEnvelopeTaskPrompt:
    @pytest.mark.parametrize("action", ["develop", "review"])
    def test_names_repository_issue_and_action(self, action) -> None:
        prompt = TaskEnvelope.model_validate(_envelope_dict(action=action)).task_prompt()
        assert REPOSITORY in prompt
        assert ISSUE_URL in prompt
        assert action in prompt

    def test_user_message_appended_only_when_set(self) -> None:
        base = TaskEnvelope.model_validate(_envelope_dict()).task_prompt()
        extended = TaskEnvelope.model_validate(_envelope_dict(user_message="Keep the diff small.")).task_prompt()
        assert extended.startswith(base)
        assert extended.endswith("Keep the diff small.")
        assert "Additional instructions" not in base
