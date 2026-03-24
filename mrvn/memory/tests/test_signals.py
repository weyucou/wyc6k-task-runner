"""Tests for memory signal handlers."""

from unittest.mock import MagicMock, patch

from agents.models import Agent
from channels.models import Channel, ChannelType, ChatRoom
from django.contrib.auth import get_user_model
from django.test import TestCase

from memory.models import Message, MessageRole, Session

User = get_user_model()


class MessagePostSaveSignalTests(TestCase):
    """Tests for handle_message_post_save signal handler."""

    def setUp(self) -> None:
        self.user = User.objects.create_user(username="testuser", password="pass")  # noqa: S106
        self.agent = Agent.objects.create(name="Test Agent")
        self.channel = Channel.objects.create(
            name="test-channel",
            channel_type=ChannelType.TELEGRAM,
            owner=self.user,
        )
        self.chat_room = ChatRoom.objects.create(
            channel=self.channel,
            platform_chat_id="chat-123",
        )
        self.session_with_agent = Session.objects.create(
            chat_room=self.chat_room,
            agent=self.agent,
        )
        self.session_without_agent = Session.objects.create(
            chat_room=self.chat_room,
            agent=None,
        )

    @patch("memory.signals.memory_search_service.index_message")
    def test_new_user_message_with_agent_triggers_indexing(self, mock_index: MagicMock) -> None:
        """New USER message on a session with an agent calls index_message."""
        msg = Message.objects.create(
            session=self.session_with_agent,
            role=MessageRole.USER,
            content="Hello world",
        )
        mock_index.assert_called_once_with(msg, self.agent.pk)

    @patch("memory.signals.memory_search_service.index_message")
    def test_new_assistant_message_with_agent_triggers_indexing(self, mock_index: MagicMock) -> None:
        """New ASSISTANT message on a session with an agent calls index_message."""
        msg = Message.objects.create(
            session=self.session_with_agent,
            role=MessageRole.ASSISTANT,
            content="Hi there",
        )
        mock_index.assert_called_once_with(msg, self.agent.pk)

    @patch("memory.signals.memory_search_service.index_message")
    def test_update_does_not_trigger_indexing(self, mock_index: MagicMock) -> None:
        """Saving an existing message (update) does not call index_message."""
        msg = Message.objects.create(
            session=self.session_with_agent,
            role=MessageRole.USER,
            content="Original",
        )
        mock_index.reset_mock()
        msg.content = "Updated"
        msg.save()
        mock_index.assert_not_called()

    @patch("memory.signals.memory_search_service.index_message")
    def test_message_without_agent_skips_indexing(self, mock_index: MagicMock) -> None:
        """New message on a session without an agent does not call index_message."""
        Message.objects.create(
            session=self.session_without_agent,
            role=MessageRole.USER,
            content="No agent here",
        )
        mock_index.assert_not_called()

    @patch("memory.signals.memory_search_service.index_message")
    def test_tool_message_skips_indexing(self, mock_index: MagicMock) -> None:
        """New TOOL message is not indexed."""
        Message.objects.create(
            session=self.session_with_agent,
            role=MessageRole.TOOL,
            content="Tool output",
        )
        mock_index.assert_not_called()

    @patch("memory.signals.memory_search_service.index_message")
    def test_system_message_skips_indexing(self, mock_index: MagicMock) -> None:
        """New SYSTEM message is not indexed."""
        Message.objects.create(
            session=self.session_with_agent,
            role=MessageRole.SYSTEM,
            content="System prompt",
        )
        mock_index.assert_not_called()
