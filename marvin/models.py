# Copyright (c) 2026 MNKT
import re
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"  # Claude on Amazon Bedrock; model_name is the Bedrock model ID
    GEMINI = "gemini"
    OLLAMA = "ollama"
    VLLM = "vllm"


class ToolProfile(StrEnum):
    MINIMAL = "minimal"
    CODING = "coding"
    MESSAGING = "messaging"
    FULL = "full"


class TaskAction(StrEnum):
    DEVELOP = "develop"
    REVIEW = "review"


_ACTION_TASKS = {
    TaskAction.DEVELOP: "Implement the issue and open a pull request for it.",
    TaskAction.REVIEW: "Review the pull request for the issue and post your review on it.",
}
_REPOSITORY_PATTERN = r"^[A-Za-z0-9-]+/[A-Za-z0-9._-]+$"  # owner/name
_GITHUB_ISSUE_URL = re.compile(r"https://github\.com/(?P<repository>[^/]+/[^/]+)/(?:issues|pull)/\d+")


class AgentConfig(BaseModel):
    """Runner-internal agent configuration; TaskEnvelope.agent_config() builds it from the wire `agent` block."""

    name: str
    description: str = ""
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model_name: str = "claude-sonnet-5-5"
    base_url: str = ""
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    is_active: bool = True
    rate_limit_enabled: bool = False
    rate_limit_rpm: int = 60
    tool_profile: ToolProfile = ToolProfile.FULL
    tools_allow: list[str] = Field(default_factory=list)
    tools_deny: list[str] = Field(default_factory=list)
    memory_search_enabled: bool = True
    memory_search_config: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    api_key: str | None = None  # Provider API key

    def get_allowed_tools(self, available_tools: list[str]) -> list[str]:
        """Calculate which tools this agent can use."""
        profile_tools = self._get_profile_tools(available_tools)
        allowed = set(profile_tools)
        if self.tools_allow:
            allowed.update(self.tools_allow)
        if self.tools_deny:
            allowed -= set(self.tools_deny)
        return [t for t in available_tools if t in allowed]

    def _get_profile_tools(self, available_tools: list[str]) -> list[str]:
        if self.tool_profile == ToolProfile.MINIMAL:
            return []
        if self.tool_profile == ToolProfile.CODING:
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
                "github_issue",
                "github_pr",
                "askcc_run",
            }
            return [t for t in available_tools if t in coding_tools]
        if self.tool_profile == ToolProfile.MESSAGING:
            messaging_prefixes = ("send", "message", "notify", "email", "slack", "telegram")
            return [t for t in available_tools if any(t.startswith(p) for p in messaging_prefixes)]
        return available_tools  # FULL


class ProjectContextConfig(BaseModel):
    """Per-customer project goals and SOPs (sourced from S3, not DB)."""

    customer_id: str
    project_id: str
    goals_markdown: str = ""
    sops_snapshot: dict[str, str] = Field(default_factory=dict)
    s3_prefix: str


class AgentSpec(BaseModel):
    """The envelope's `agent` block. Unknown fields, such as AgentConfig's runner-internal ones, are ignored."""

    provider: LLMProvider
    model_name: str
    tool_profile: ToolProfile
    max_tokens: int = 4096
    tools_allow: list[str] = Field(default_factory=list)
    tools_deny: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _reject_api_key(cls, data: Any) -> Any:
        if isinstance(data, dict) and "api_key" in data:
            raise ValueError("agent.api_key is not accepted; keys come from *_secret_id or the runner environment")
        return data


class TaskEnvelope(BaseModel):
    """SQS message payload: the v1 contract in weyucou/wyc6k-spec docs/architecture.md §TaskEnvelope.

    The fields through user_message are the wire contract, in contract order; the rest are runner-internal.
    """

    # Backstop: str(ValidationError) omits input values, so a logged traceback cannot echo agent.api_key.
    model_config = ConfigDict(hide_input_in_errors=True)

    schema_version: Literal[1]
    task_id: str
    customer_id: str
    task_source_id: int
    repository: str = Field(pattern=_REPOSITORY_PATTERN)
    issue_url: str
    action: TaskAction
    agent: AgentSpec
    context_s3_prefix: str
    # Estimated runtime, used by the control plane (wyc6k-task-manager) to pick a
    # compute backend. The runtime itself ignores it.
    duration_hint_seconds: int = 3600
    max_tool_iterations: int = 10
    github_token_secret_id: str | None = None
    anthropic_api_key_secret_id: str | None = None
    user_message: str = ""  # optional extra instruction, appended to task_prompt()

    session_id: str = ""  # defaults to task_id
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    enable_tools: bool = True

    @field_validator("issue_url")
    @classmethod
    def _issue_url_in_repository(cls, issue_url: str, info: ValidationInfo) -> str:
        repository = info.data.get("repository")
        if repository is None:  # missing or invalid; that error already names the field
            return issue_url
        match = _GITHUB_ISSUE_URL.fullmatch(issue_url)
        if match is None or match["repository"].lower() != repository.lower():
            raise ValueError("issue_url must be https://github.com/<repository>/issues/<n> or /pull/<n>")
        return issue_url

    @model_validator(mode="after")
    def _derive_session_id(self) -> Self:
        if not self.session_id:
            self.session_id = self.task_id
        return self

    def agent_config(self, api_key: str | None) -> AgentConfig:
        """Build the runner's AgentConfig from the agent block and the resolved API key."""
        # The name keys the rate limiter.
        return AgentConfig(name=f"{self.repository}#{self.action}", api_key=api_key, **self.agent.model_dump())

    def task_prompt(self) -> str:
        """Build the agent's task from repository, issue_url and action, then user_message when set."""
        prompt = (
            f"Repository: {self.repository}\n"
            f"Issue: {self.issue_url}\n"
            f"Action: {self.action}\n\n"
            f"{_ACTION_TASKS[self.action]}"
        )
        if self.user_message:
            prompt += f"\n\nAdditional instructions:\n{self.user_message}"
        return prompt
