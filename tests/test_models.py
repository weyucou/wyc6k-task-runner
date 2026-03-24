"""Tests for marvin.models — AgentConfig, TaskEnvelope, ToolProfile."""

import pytest
from pydantic import ValidationError

from marvin.models import AgentConfig, LLMProvider, TaskEnvelope, ToolProfile


class TestAgentConfig:
    def test_defaults(self) -> None:
        agent = AgentConfig(name="test-agent")
        assert agent.provider == LLMProvider.ANTHROPIC
        assert agent.model_name == "claude-sonnet-4-20250514"
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


class TestTaskEnvelope:
    def _make_agent(self) -> AgentConfig:
        return AgentConfig(name="test-agent")

    def test_valid_envelope(self) -> None:
        agent = self._make_agent()
        envelope = TaskEnvelope(
            task_id="t-001",
            customer_id="c-001",
            session_id="s-001",
            agent=agent,
            s3_context_prefix="s3://bucket/customer/projects/repo/",
            user_message="Hello",
        )
        assert envelope.task_id == "t-001"
        assert envelope.enable_tools is True
        assert envelope.max_tool_iterations == 10
        assert envelope.conversation_history == []

    def test_requires_task_id(self) -> None:
        with pytest.raises(ValidationError):
            TaskEnvelope(  # type: ignore[call-arg]
                customer_id="c-001",
                session_id="s-001",
                agent=self._make_agent(),
                s3_context_prefix="s3://bucket/prefix",
                user_message="Hi",
            )

    def test_model_validate_from_dict(self) -> None:
        data = {
            "task_id": "t-002",
            "customer_id": "c-002",
            "session_id": "s-002",
            "agent": {
                "name": "agent",
                "provider": "anthropic",
                "model_name": "claude-sonnet-4-20250514",
            },
            "s3_context_prefix": "s3://bucket/prefix",
            "user_message": "Test",
            "conversation_history": [{"role": "user", "content": "Previous message"}],
        }
        envelope = TaskEnvelope.model_validate(data)
        assert envelope.task_id == "t-002"
        assert len(envelope.conversation_history) == 1
        assert isinstance(envelope.agent, AgentConfig)
