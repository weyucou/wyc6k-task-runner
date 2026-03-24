import logging
from enum import StrEnum

from commons.models import TimestampedModel
from django.db import models

logger = logging.getLogger(__name__)


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    VLLM = "vllm"


class ToolProfile(StrEnum):
    """Pre-defined tool access profiles."""

    MINIMAL = "minimal"  # No tools by default
    CODING = "coding"  # Development tools (read, write, exec)
    MESSAGING = "messaging"  # Communication tools
    FULL = "full"  # All available tools


class Agent(TimestampedModel):
    """An AI agent configuration."""

    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)

    # LLM Configuration
    provider = models.CharField(
        max_length=20,
        choices=[(p.value, p.name.title()) for p in LLMProvider],
        default=LLMProvider.ANTHROPIC.value,
    )
    model_name = models.CharField(max_length=100, default="claude-sonnet-4-20250514")

    # Base URL for local LLMs (Ollama, vLLM) or custom API endpoints
    base_url = models.URLField(blank=True, help_text="Required for Ollama/vLLM providers")

    # System prompt
    system_prompt = models.TextField(blank=True)

    # Agent behavior settings
    temperature = models.FloatField(default=0.7)
    max_tokens = models.IntegerField(default=4096)

    is_active = models.BooleanField(default=True)

    # Rate limiting to avoid external API throttling
    rate_limit_enabled = models.BooleanField(
        default=False,
        help_text="Enable rate limiting for API calls",
    )
    rate_limit_rpm = models.PositiveIntegerField(
        default=60,
        help_text="Maximum requests per minute (0 = unlimited)",
    )

    # Tool access control (inspired by OpenClaw)
    tool_profile = models.CharField(
        max_length=20,
        choices=[(p.value, p.name.title()) for p in ToolProfile],
        default=ToolProfile.FULL.value,
        help_text="Pre-defined tool access profile",
    )
    tools_allow = models.JSONField(
        default=list,
        blank=True,
        help_text="List of tool names to explicitly allow (overrides profile)",
    )
    tools_deny = models.JSONField(
        default=list,
        blank=True,
        help_text="List of tool names to explicitly deny (overrides allow)",
    )

    # Memory search configuration
    memory_search_enabled = models.BooleanField(
        default=True,
        help_text="Enable memory search tool for this agent",
    )
    memory_search_config = models.JSONField(
        default=dict,
        blank=True,
        help_text="Memory search config: max_results, min_score, hybrid_weights, etc.",
    )

    # Additional config (stop sequences, etc.)
    config = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_datetime"]

    def __str__(self) -> str:
        return f"{self.name} ({self.model_name})"

    def get_allowed_tools(self, available_tools: list[str]) -> list[str]:
        """Calculate which tools this agent can use.

        Tool resolution order (most restrictive wins):
        1. Start with profile-based tools
        2. Add explicitly allowed tools
        3. Remove explicitly denied tools

        Args:
            available_tools: List of all available tool names.

        Returns:
            List of tool names the agent is allowed to use.
        """
        profile_tools = self._get_profile_tools(available_tools)

        # Add explicitly allowed tools
        allowed = set(profile_tools)
        if self.tools_allow:
            allowed.update(self.tools_allow)

        # Remove explicitly denied tools (deny wins over allow)
        if self.tools_deny:
            allowed -= set(self.tools_deny)

        # Only return tools that actually exist
        return [t for t in available_tools if t in allowed]

    def _get_profile_tools(self, available_tools: list[str]) -> list[str]:
        """Get tools based on the tool profile.

        Args:
            available_tools: List of all available tool names.

        Returns:
            List of tool names for the profile.
        """
        if self.tool_profile == ToolProfile.MINIMAL.value:
            return []
        if self.tool_profile == ToolProfile.CODING.value:
            # Explicit set of tools for coding/agent-development workflows
            coding_tools = {
                "read",
                "write",
                "edit",
                "apply_patch",
                "exec",
                "process",
                "web_fetch",
                "web_search",
                "browser_fetch",
                "sessions_spawn",
                "sessions_send",
                "image",
                "memory_store",
                "memory_retrieve",
                "memory_search",
                "s3_memory_write",
            }
            return [t for t in available_tools if t in coding_tools]
        if self.tool_profile == ToolProfile.MESSAGING.value:
            # Return messaging-related tools
            messaging_prefixes = ("send", "message", "notify", "email", "slack", "telegram")
            return [t for t in available_tools if any(t.startswith(p) for p in messaging_prefixes)]
        # FULL profile - return all tools
        return available_tools


class AgentCredential(TimestampedModel):
    """API credentials for an agent's LLM provider."""

    agent = models.OneToOneField(
        Agent,
        on_delete=models.CASCADE,
        related_name="credential",
    )

    # Encrypted API key
    encrypted_api_key = models.TextField()

    class Meta:
        verbose_name = "Agent Credential"
        verbose_name_plural = "Agent Credentials"


class Tool(TimestampedModel):
    """A tool that an agent can use."""

    name = models.CharField(max_length=100, unique=True)
    description = models.TextField()

    # Tool schema (JSON Schema for parameters)
    input_schema = models.JSONField(default=dict)

    # Python path to the tool implementation
    handler_path = models.CharField(max_length=255)

    is_active = models.BooleanField(default=True)

    # Security: which contexts can use this tool
    allow_in_groups = models.BooleanField(default=False)
    require_approval = models.BooleanField(default=False)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class AgentTool(TimestampedModel):
    """Many-to-many relationship between agents and tools with config."""

    agent = models.ForeignKey(
        Agent,
        on_delete=models.CASCADE,
        related_name="agent_tools",
    )
    tool = models.ForeignKey(
        Tool,
        on_delete=models.CASCADE,
        related_name="agent_tools",
    )

    is_enabled = models.BooleanField(default=True)

    # Tool-specific configuration for this agent
    config = models.JSONField(default=dict, blank=True)

    class Meta:
        unique_together = [("agent", "tool")]

    def __str__(self) -> str:
        return f"{self.agent.name} - {self.tool.name}"


class ProjectContext(TimestampedModel):
    """Per-customer project goals and SOPs cached from S3, keyed by GitHub project."""

    customer = models.ForeignKey(
        "accounts.Customer",
        on_delete=models.CASCADE,
        related_name="project_contexts",
    )
    project_id = models.CharField(max_length=255, help_text="GitHub project node ID")
    goals_markdown = models.TextField(blank=True, help_text="Project goals (manually set or synced)")
    sops_snapshot = models.JSONField(default=dict, help_text="Cached SOP content from S3")
    s3_prefix = models.CharField(max_length=512, help_text="s3://bucket/customer/projects/<name>/")
    last_synced = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = [("customer", "project_id")]

    def __str__(self) -> str:
        return f"{self.customer}/{self.project_id}"
