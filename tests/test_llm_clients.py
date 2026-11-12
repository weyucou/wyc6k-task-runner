# Copyright (c) 2026 MNKT
"""Tests for LLM client request kwargs, checked against the installed provider SDKs."""

import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import AnthropicError, AsyncAnthropic
from anthropic.lib.bedrock import AsyncAnthropicBedrockMantle
from anthropic.resources.messages import AsyncMessages
from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI

from marvin.llm.anthropic_client import AnthropicClient
from marvin.llm.base import LLMMessage, StopReason
from marvin.llm.bedrock_client import BedrockClient
from marvin.llm.factory import create_llm_client
from marvin.llm.gemini_client import GeminiClient
from marvin.llm.openai_client import OpenAIClient
from marvin.models import AgentConfig, LLMProvider
from marvin.runner import AgentRunner
from marvin.tools import ToolRegistry
from marvin.tools.builtin import CalculatorTool

SAMPLING_PARAMS = {"temperature", "top_p", "top_k"}
MESSAGES = [LLMMessage.user("What is 2 + 2?")]


def _assert_sdk_accepts(sdk_method: Any, kwargs: dict[str, Any]) -> None:
    """Raise TypeError when the installed SDK method has no parameter for a request kwarg."""
    inspect.signature(sdk_method).bind(**kwargs)


def _anthropic_message(block: SimpleNamespace, stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[block],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        model="claude-sonnet-5-5",
    )


def _anthropic_client(*replies: SimpleNamespace) -> tuple[AnthropicClient, AsyncMock]:
    create = AsyncMock(side_effect=list(replies))
    client = AnthropicClient(api_key="x")
    client._client = MagicMock()
    client._client.messages.create = create
    return client, create


ANTHROPIC_TEXT_REPLY = _anthropic_message(SimpleNamespace(type="text", text="4"), "end_turn")
ANTHROPIC_TOOL_REPLY = _anthropic_message(
    SimpleNamespace(type="tool_use", id="toolu_1", name="calculator", input={"expression": "2 + 2"}),
    "tool_use",
)


class TestAnthropicClient:
    @pytest.mark.asyncio
    async def test_request_omits_sampling_params(self) -> None:
        client, create = _anthropic_client(ANTHROPIC_TEXT_REPLY)
        tools = [CalculatorTool().to_anthropic_format()]

        response = await client.generate(MESSAGES, system_prompt="Be brief.", tools=tools, temperature=0.3)

        assert response.stop_reason == StopReason.END_TURN
        kwargs = create.call_args.kwargs
        assert not SAMPLING_PARAMS & kwargs.keys()
        _assert_sdk_accepts(AsyncAnthropic(api_key="x").messages.create, kwargs)

    @pytest.mark.asyncio
    async def test_generate_with_tools_omits_sampling_params(self) -> None:
        # One tool round, then the max-iterations final call: covers both generate() call sites.
        client, create = _anthropic_client(ANTHROPIC_TOOL_REPLY, ANTHROPIC_TEXT_REPLY)
        registry = ToolRegistry()
        registry.register(CalculatorTool())

        response, _ = await client.generate_with_tools(
            MESSAGES,
            registry.to_anthropic_tools(),
            registry,
            temperature=0.3,
            max_iterations=1,
        )

        assert response.stop_reason == StopReason.END_TURN
        assert create.await_count == 2
        for call in create.call_args_list:
            assert not SAMPLING_PARAMS & call.kwargs.keys()

    def test_default_model(self, monkeypatch) -> None:
        monkeypatch.delenv("DEFAULT_ANTHROPIC_MODEL", raising=False)
        assert AnthropicClient().model == "claude-sonnet-5-5"


BEDROCK_MODEL = "anthropic.claude-sonnet-5-5"
# Env vars that reach the Bedrock client or the AWS credential chain. Cleared, so tests never see real AWS config.
AWS_ENV_NAMES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_BEARER_TOKEN_BEDROCK",
    "ANTHROPIC_AWS_API_KEY",
    "ANTHROPIC_BEDROCK_MANTLE_BASE_URL",
    "ANTHROPIC_API_KEY",
    "BEDROCK_AWS_REGION",
    "BEDROCK_AWS_PROFILE",
)


class TestBedrockClient:
    @pytest.fixture(autouse=True)
    def _aws_env(self, monkeypatch, tmp_path) -> None:
        """Env credentials only: empty AWS config files, no region, no instance-metadata lookup."""
        for name in AWS_ENV_NAMES:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")

    @pytest.mark.parametrize("provider", ["bedrock", LLMProvider.BEDROCK], ids=["str", "enum"])
    def test_factory_drops_api_key_and_base_url(self, provider) -> None:
        client = create_llm_client(
            provider, api_key="sk-ant-x", base_url="https://api.anthropic.com", model=BEDROCK_MODEL
        )
        assert client.get_provider_name() == "bedrock"
        assert client.api_key is None
        assert client.base_url is None
        assert client.model == BEDROCK_MODEL

    @pytest.mark.parametrize(
        ("bedrock_region", "region"),
        [("us-west-2", "us-west-2"), ("", "ap-northeast-1")],
        ids=["bedrock-region", "falls-back-to-default-region"],
    )
    def test_sdk_client(self, monkeypatch, bedrock_region, region) -> None:
        monkeypatch.setenv("BEDROCK_AWS_REGION", bedrock_region)
        monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-northeast-1")
        client = BedrockClient(api_key="sk-ant-x", base_url="https://api.anthropic.com", model=BEDROCK_MODEL)
        sdk_client = client._get_client()
        assert isinstance(sdk_client, AsyncAnthropicBedrockMantle)
        assert sdk_client.aws_region == region
        assert str(sdk_client.base_url) == f"https://bedrock-mantle.{region}.api.aws/anthropic/"
        assert sdk_client.api_key is None  # SigV4-signed; the Anthropic key is never sent

    def test_profile_forwarded(self, monkeypatch, tmp_path) -> None:
        profile = "[profile bedrock-dev]\naws_access_key_id = profile-key\naws_secret_access_key = profile-secret\n"
        (tmp_path / "config").write_text(profile)
        monkeypatch.setenv("BEDROCK_AWS_REGION", "us-west-2")
        monkeypatch.setenv("BEDROCK_AWS_PROFILE", "bedrock-dev")
        assert BedrockClient(model=BEDROCK_MODEL)._get_client().aws_profile == "bedrock-dev"

    def test_bearer_token_needs_no_aws_credentials(self, monkeypatch) -> None:
        monkeypatch.setenv("BEDROCK_AWS_REGION", "us-west-2")
        monkeypatch.delenv("AWS_ACCESS_KEY_ID")
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY")
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key")
        assert BedrockClient(model=BEDROCK_MODEL)._get_client().api_key == "bedrock-api-key"

    # generate() turns request errors into an "Error: ..." reply, and entrypoint.py exits 0 on a reply.
    # Setup errors must raise from generate() instead, so the task exits 1.

    @pytest.mark.asyncio
    async def test_missing_credentials_fail_the_task(self, monkeypatch) -> None:
        monkeypatch.setenv("BEDROCK_AWS_REGION", "us-west-2")
        monkeypatch.delenv("AWS_ACCESS_KEY_ID")
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY")
        with pytest.raises(RuntimeError, match="No AWS credentials for Bedrock"):
            await BedrockClient(model=BEDROCK_MODEL).generate(MESSAGES)

    @pytest.mark.asyncio
    async def test_profile_ignores_env_credentials(self, monkeypatch, tmp_path) -> None:
        # The local-runner forwards LocalStack's fake env credentials; a Bedrock profile must not fall back to them.
        (tmp_path / "config").write_text("[profile bedrock-dev]\n")
        monkeypatch.setenv("BEDROCK_AWS_REGION", "us-west-2")
        monkeypatch.setenv("BEDROCK_AWS_PROFILE", "bedrock-dev")
        with pytest.raises(RuntimeError, match="No AWS credentials for Bedrock"):
            await BedrockClient(model=BEDROCK_MODEL).generate(MESSAGES)

    @pytest.mark.asyncio
    async def test_missing_region_fails_the_task(self) -> None:
        with pytest.raises(AnthropicError, match="No AWS region"):
            await BedrockClient(model=BEDROCK_MODEL).generate(MESSAGES)

    @pytest.mark.asyncio
    async def test_request_passes_model_through(self, monkeypatch) -> None:
        monkeypatch.setenv("BEDROCK_AWS_REGION", "us-west-2")
        client = BedrockClient(model=BEDROCK_MODEL)
        create = AsyncMock(return_value=ANTHROPIC_TEXT_REPLY)
        tools = [CalculatorTool().to_anthropic_format()]

        with patch.object(AsyncMessages, "create", create):
            response = await client.generate(MESSAGES, tools=tools, temperature=0.3)

        assert response.content == "4"
        kwargs = create.call_args.kwargs
        assert kwargs["model"] == BEDROCK_MODEL
        assert not SAMPLING_PARAMS & kwargs.keys()
        _assert_sdk_accepts(client._get_client().messages.create, kwargs)

    @pytest.mark.asyncio
    async def test_runner_tool_round_trip(self, monkeypatch) -> None:
        monkeypatch.setenv("BEDROCK_AWS_REGION", "us-west-2")
        agent = AgentConfig(name="bedrock-agent", provider=LLMProvider.BEDROCK, model_name=BEDROCK_MODEL)
        runner = AgentRunner(agent=agent, register_builtins=False)
        runner.register_tool(CalculatorTool())
        create = AsyncMock(side_effect=[ANTHROPIC_TOOL_REPLY, ANTHROPIC_TEXT_REPLY])

        with patch.object(AsyncMessages, "create", create):
            text, _ = await runner.chat("What is 2 + 2?")

        assert text == "4"
        tool_use_call, final_call = (call.kwargs for call in create.call_args_list)
        assert tool_use_call["tools"] == runner.registry.to_anthropic_tools()
        tool_result = {"type": "tool_result", "tool_use_id": "toolu_1", "content": "4"}
        assert final_call["messages"][-1]["content"] == [tool_result]


def _gemini_client() -> tuple[GeminiClient, AsyncMock]:
    generate_content = AsyncMock(
        return_value=SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[SimpleNamespace(text="4", function_call=None)]),
                    finish_reason="STOP",
                )
            ],
            usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
        )
    )
    client = GeminiClient(api_key="x")
    client._client = MagicMock()
    client._client.aio.models.generate_content = generate_content
    return client, generate_content


class TestGeminiClient:
    @pytest.mark.asyncio
    async def test_tools_passed_in_config(self) -> None:
        client, generate_content = _gemini_client()
        tools = [CalculatorTool().to_gemini_format()]

        response = await client.generate(MESSAGES, system_prompt="Be brief.", tools=tools, temperature=0.3)

        assert response.stop_reason == StopReason.END_TURN
        kwargs = generate_content.call_args.kwargs
        assert "tools" not in kwargs
        assert kwargs["config"]["tools"][0]["function_declarations"][0]["name"] == "calculator"
        assert kwargs["config"]["temperature"] == 0.3
        _assert_sdk_accepts(genai.Client(api_key="x").aio.models.generate_content, kwargs)
        genai_types.GenerateContentConfig.model_validate(kwargs["config"])

    def test_default_model(self, monkeypatch) -> None:
        monkeypatch.delenv("DEFAULT_GEMINI_MODEL", raising=False)
        assert GeminiClient().model == "gemini-3.6-flash"


def _openai_client() -> tuple[OpenAIClient, AsyncMock]:
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="4", tool_calls=None), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            model="served-model",
        )
    )
    client = OpenAIClient(api_key="x", base_url="http://localhost:8000/v1", model="served-model")
    client._client = MagicMock()
    client._client.chat.completions.create = create
    return client, create


class TestOpenAIClient:
    @pytest.mark.asyncio
    async def test_request_kwargs(self) -> None:
        client, create = _openai_client()
        tools = [CalculatorTool().to_openai_format()]

        response = await client.generate(MESSAGES, tools=tools, temperature=0.3, max_tokens=256)

        assert response.stop_reason == StopReason.END_TURN
        kwargs = create.call_args.kwargs
        assert kwargs.keys() == {"model", "messages", "max_tokens", "temperature", "tools", "tool_choice"}
        assert kwargs["temperature"] == 0.3
        assert kwargs["tool_choice"] == "auto"
        _assert_sdk_accepts(AsyncOpenAI(api_key="x").chat.completions.create, kwargs)
