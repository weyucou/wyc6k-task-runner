"""Tests for AgentRunner transcript replay feature."""

from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from agents.llm.base import LLMMessage, LLMResponse, MessageRole, StopReason
from agents.runner import AgentRunner, session_messages_to_llm_messages


def _make_record(
    role: str, content: str, *, tool_call_id: str = "", tool_name: str = "", raw_data: dict | None = None
) -> MagicMock:
    """Build a mock memory.Message record."""
    record = MagicMock()
    record.role = role
    record.content = content
    record.tool_call_id = tool_call_id
    record.tool_name = tool_name
    record.raw_data = raw_data or {}
    return record


class SessionMessagesToLLMMessagesTests(IsolatedAsyncioTestCase):
    """Tests for the session_messages_to_llm_messages conversion function."""

    def test_user_message_converted(self) -> None:
        records = [_make_record("user", "Hello")]
        result = session_messages_to_llm_messages(records)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].role, MessageRole.USER)
        self.assertEqual(result[0].content, "Hello")

    def test_assistant_message_converted(self) -> None:
        records = [_make_record("assistant", "Hi there")]
        result = session_messages_to_llm_messages(records)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].role, MessageRole.ASSISTANT)
        self.assertEqual(result[0].content, "Hi there")
        self.assertIsNone(result[0].tool_calls)

    def test_assistant_message_with_tool_calls_reconstructed(self) -> None:
        raw = {"tool_calls": [{"id": "tc_1", "name": "calculator", "arguments": {"expression": "2+2"}}]}
        records = [_make_record("assistant", "", raw_data=raw)]
        result = session_messages_to_llm_messages(records)
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result[0].tool_calls)
        tc = result[0].tool_calls[0]
        self.assertEqual(tc.id, "tc_1")
        self.assertEqual(tc.name, "calculator")
        self.assertEqual(tc.arguments, {"expression": "2+2"})

    def test_tool_result_message_converted(self) -> None:
        records = [_make_record("tool", "4", tool_call_id="tc_1", tool_name="calculator")]
        result = session_messages_to_llm_messages(records)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].role, MessageRole.TOOL)
        self.assertEqual(result[0].content, "4")
        self.assertEqual(result[0].tool_call_id, "tc_1")
        self.assertEqual(result[0].name, "calculator")

    def test_system_messages_skipped(self) -> None:
        records = [
            _make_record("system", "You are a helper"),
            _make_record("user", "Hi"),
        ]
        result = session_messages_to_llm_messages(records)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].role, MessageRole.USER)

    def test_empty_tool_name_maps_to_none(self) -> None:
        records = [_make_record("tool", "result", tool_call_id="tc_2", tool_name="")]
        result = session_messages_to_llm_messages(records)
        self.assertIsNone(result[0].name)

    def test_chronological_order_preserved(self) -> None:
        records = [
            _make_record("user", "first"),
            _make_record("assistant", "second"),
            _make_record("user", "third"),
        ]
        result = session_messages_to_llm_messages(records)
        self.assertEqual([m.content for m in result], ["first", "second", "third"])


def _make_agent(context_window_messages: int = 0) -> MagicMock:
    """Build a mock Agent with the given context_window_messages value."""
    agent = MagicMock()
    agent.id = 1
    agent.context_window_messages = context_window_messages
    agent.system_prompt = ""
    agent.temperature = 0.7
    agent.max_tokens = 4096
    agent.provider = "anthropic"
    agent.rate_limit_enabled = False
    return agent


def _make_llm_response(content: str = "OK") -> LLMResponse:
    return LLMResponse(content=content, stop_reason=StopReason.END_TURN)


class AgentRunnerTranscriptTests(IsolatedAsyncioTestCase):
    """Tests for AgentRunner session transcript injection in run()."""

    async def test_no_transcript_when_context_window_messages_is_zero(self) -> None:
        """When context_window_messages=0, run() must not prepend any history."""
        agent = _make_agent(context_window_messages=0)
        runner = AgentRunner(agent, register_builtins=False)

        user_msg = LLMMessage.user("Hello")
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=_make_llm_response())
        runner._client = mock_client

        with patch.object(runner, "_load_session_transcript", new_callable=AsyncMock) as mock_load:
            session = MagicMock()
            await runner.run([user_msg], session=session, enable_tools=False)
            mock_load.assert_not_called()

        # Verify the client received exactly the original message
        called_messages = mock_client.generate.call_args[0][0]
        self.assertEqual(len(called_messages), 1)
        self.assertEqual(called_messages[0].content, "Hello")

    async def test_transcript_injected_before_user_message(self) -> None:
        """Prior session messages are prepended to the incoming message list."""
        agent = _make_agent(context_window_messages=5)
        runner = AgentRunner(agent, register_builtins=False)

        prior = [LLMMessage.user("prior message")]
        user_msg = LLMMessage.user("new message")
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=_make_llm_response())
        runner._client = mock_client

        with patch.object(runner, "_load_session_transcript", new_callable=AsyncMock, return_value=prior):
            session = MagicMock()
            await runner.run([user_msg], session=session, enable_tools=False)

        called_messages = mock_client.generate.call_args[0][0]
        self.assertEqual(len(called_messages), 2)
        self.assertEqual(called_messages[0].content, "prior message")
        self.assertEqual(called_messages[1].content, "new message")

    async def test_no_session_no_transcript(self) -> None:
        """When session=None, no transcript loading occurs even if configured."""
        agent = _make_agent(context_window_messages=5)
        runner = AgentRunner(agent, register_builtins=False)

        user_msg = LLMMessage.user("Hello")
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=_make_llm_response())
        runner._client = mock_client

        with patch.object(runner, "_load_session_transcript", new_callable=AsyncMock) as mock_load:
            await runner.run([user_msg], session=None, enable_tools=False)
            mock_load.assert_not_called()

    async def test_empty_transcript_does_not_change_messages(self) -> None:
        """When the session has no messages, the message list is unchanged."""
        agent = _make_agent(context_window_messages=5)
        runner = AgentRunner(agent, register_builtins=False)

        user_msg = LLMMessage.user("Hello")
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=_make_llm_response())
        runner._client = mock_client

        with patch.object(runner, "_load_session_transcript", new_callable=AsyncMock, return_value=[]):
            session = MagicMock()
            await runner.run([user_msg], session=session, enable_tools=False)

        called_messages = mock_client.generate.call_args[0][0]
        self.assertEqual(len(called_messages), 1)
        self.assertEqual(called_messages[0].content, "Hello")

    async def test_load_session_transcript_calls_correct_n(self) -> None:
        """_load_session_transcript is called with agent.context_window_messages as N."""
        agent = _make_agent(context_window_messages=7)
        runner = AgentRunner(agent, register_builtins=False)

        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=_make_llm_response())
        runner._client = mock_client

        with patch.object(runner, "_load_session_transcript", new_callable=AsyncMock, return_value=[]) as mock_load:
            session = MagicMock()
            await runner.run([LLMMessage.user("hi")], session=session, enable_tools=False)
            mock_load.assert_awaited_once_with(session, 7)
