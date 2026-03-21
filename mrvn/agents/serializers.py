"""DRF serializers for the agents app."""

from typing import Any

from rest_framework import serializers

from agents.models import Agent, AgentTool, LLMProvider, Tool


class ToolSerializer(serializers.ModelSerializer):
    """Serializer for Tool model."""

    class Meta:
        model = Tool
        fields = [
            "id",
            "name",
            "description",
            "input_schema",
            "is_active",
            "allow_in_groups",
            "require_approval",
            "created_datetime",
            "updated_datetime",
        ]
        read_only_fields = ["id", "created_datetime", "updated_datetime"]


class AgentToolSerializer(serializers.ModelSerializer):
    """Serializer for AgentTool model."""

    tool_name = serializers.CharField(source="tool.name", read_only=True)
    tool_description = serializers.CharField(source="tool.description", read_only=True)

    class Meta:
        model = AgentTool
        fields = [
            "id",
            "tool",
            "tool_name",
            "tool_description",
            "is_enabled",
            "config",
        ]
        read_only_fields = ["id", "tool_name", "tool_description"]


class AgentListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for agent listing."""

    class Meta:
        model = Agent
        fields = [
            "id",
            "name",
            "description",
            "provider",
            "model_name",
            "is_active",
            "created_datetime",
            "updated_datetime",
        ]
        read_only_fields = ["id", "created_datetime", "updated_datetime"]


class AgentDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer for agent with full configuration."""

    agent_tools = AgentToolSerializer(many=True, read_only=True)
    provider_choices = serializers.SerializerMethodField()

    class Meta:
        model = Agent
        fields = [
            "id",
            "name",
            "description",
            "provider",
            "provider_choices",
            "model_name",
            "base_url",
            "system_prompt",
            "temperature",
            "max_tokens",
            "is_active",
            "rate_limit_enabled",
            "rate_limit_rpm",
            "tool_profile",
            "tools_allow",
            "tools_deny",
            "memory_search_enabled",
            "memory_search_config",
            "config",
            "agent_tools",
            "created_datetime",
            "updated_datetime",
        ]
        read_only_fields = [
            "id",
            "provider_choices",
            "agent_tools",
            "created_datetime",
            "updated_datetime",
        ]

    def get_provider_choices(self, _obj: Agent) -> list[dict[str, str]]:
        """Return available provider choices."""
        return [{"value": p.value, "label": p.name.title()} for p in LLMProvider]

    def validate_temperature(self, value: float) -> float:
        """Validate temperature is within valid range."""
        max_temperature = 2.0
        if not 0.0 <= value <= max_temperature:
            msg = f"Temperature must be between 0.0 and {max_temperature}"
            raise serializers.ValidationError(msg)
        return value

    def validate_max_tokens(self, value: int) -> int:
        """Validate max_tokens is positive."""
        if value < 1:
            msg = "max_tokens must be at least 1"
            raise serializers.ValidationError(msg)
        return value


class AgentCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a new agent."""

    api_key = serializers.CharField(write_only=True, required=False, allow_blank=True)

    class Meta:
        model = Agent
        fields = [
            "name",
            "description",
            "provider",
            "model_name",
            "base_url",
            "system_prompt",
            "temperature",
            "max_tokens",
            "is_active",
            "rate_limit_enabled",
            "rate_limit_rpm",
            "tool_profile",
            "tools_allow",
            "tools_deny",
            "memory_search_enabled",
            "memory_search_config",
            "config",
            "api_key",
        ]

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        """Validate provider-specific requirements."""
        provider = attrs.get("provider", LLMProvider.ANTHROPIC.value)

        # Require base_url for local providers
        if provider in (LLMProvider.OLLAMA.value, LLMProvider.VLLM.value):
            if not attrs.get("base_url"):
                msg = f"base_url is required for {provider} provider"
                raise serializers.ValidationError({"base_url": msg})
        # Require API key for cloud providers
        elif not attrs.get("api_key"):
            msg = f"api_key is required for {provider} provider"
            raise serializers.ValidationError({"api_key": msg})

        return attrs


class ChatMessageSerializer(serializers.Serializer):
    """Serializer for chat messages."""

    role = serializers.ChoiceField(choices=["user", "assistant", "system"])
    content = serializers.CharField()


class ChatRequestSerializer(serializers.Serializer):
    """Serializer for chat API request."""

    message = serializers.CharField()
    conversation_history = ChatMessageSerializer(many=True, required=False, default=list)
    enable_tools = serializers.BooleanField(default=True)
    tool_names = serializers.ListField(child=serializers.CharField(), required=False, allow_null=True)
    system_prompt = serializers.CharField(required=False, allow_blank=True, allow_null=True)


class ChatResponseSerializer(serializers.Serializer):
    """Serializer for chat API response."""

    content = serializers.CharField()
    agent = serializers.CharField()
    model = serializers.CharField()
    stop_reason = serializers.CharField()
    input_tokens = serializers.IntegerField()
    output_tokens = serializers.IntegerField()
    tool_calls = serializers.ListField(required=False, default=list)
    history = serializers.ListField(required=False)


class MemorySearchRequestSerializer(serializers.Serializer):
    """Serializer for memory search API request."""

    query = serializers.CharField()
    search_type = serializers.ChoiceField(
        choices=["hybrid", "vector", "text"],
        default="hybrid",
    )
    max_results = serializers.IntegerField(min_value=1, max_value=10, default=5)
    session_id = serializers.IntegerField(required=False, allow_null=True)


class MemorySearchResultSerializer(serializers.Serializer):
    """Serializer for a single memory search result."""

    content = serializers.CharField()
    score = serializers.FloatField()
    source = serializers.CharField()
    message_id = serializers.IntegerField(allow_null=True)
    summary_id = serializers.IntegerField(allow_null=True)
    metadata = serializers.DictField()


class MemorySearchResponseSerializer(serializers.Serializer):
    """Serializer for memory search API response."""

    query = serializers.CharField()
    search_type = serializers.CharField()
    results = MemorySearchResultSerializer(many=True)
    count = serializers.IntegerField()
