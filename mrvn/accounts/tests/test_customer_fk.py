from django.contrib.auth import get_user_model
from django.test import TestCase

from accounts.models import Customer
from agents.models import Agent
from channels.models import Channel, ChannelType, ChatRoom
from memory.models import Session

User = get_user_model()


def make_customer(**kwargs) -> Customer:
    defaults = {
        "name": "Test Corp",
        "github_org": "testcorp",
        "s3_context_prefix": "cust_test/",
    }
    defaults.update(kwargs)
    return Customer.objects.create(**defaults)


def make_user(username="testuser") -> User:
    return User.objects.create_user(username=username, password="pass")


def make_agent(owner, customer=None) -> Agent:
    return Agent.objects.create(
        name="Test Agent",
        owner=owner,
        customer=customer,
    )


def make_channel(owner) -> Channel:
    return Channel.objects.create(
        name="Test Channel",
        channel_type=ChannelType.TELEGRAM,
        owner=owner,
    )


def make_chat_room(channel, agent=None, customer=None) -> ChatRoom:
    return ChatRoom.objects.create(
        channel=channel,
        platform_chat_id="chat_001",
        agent=agent,
        customer=customer,
    )


class AgentCustomerFKTests(TestCase):
    def setUp(self):
        self.user = make_user()
        self.customer = make_customer()

    def test_agent_customer_null_by_default(self):
        agent = make_agent(self.user)
        self.assertIsNone(agent.customer)

    def test_agent_customer_assignment(self):
        agent = make_agent(self.user, customer=self.customer)
        self.assertEqual(agent.customer, self.customer)

    def test_agent_filter_by_customer_id(self):
        make_agent(self.user, customer=self.customer)
        make_agent(self.user.__class__.objects.create_user(username="other", password="x"))
        qs = Agent.objects.filter(customer_id=self.customer.pk)
        self.assertEqual(qs.count(), 1)

    def test_agent_customer_set_null_on_delete(self):
        agent = make_agent(self.user, customer=self.customer)
        self.customer.delete()
        agent.refresh_from_db()
        self.assertIsNone(agent.customer_id)


class ChatRoomCustomerFKTests(TestCase):
    def setUp(self):
        self.user = make_user()
        self.customer = make_customer()
        self.channel = make_channel(self.user)

    def test_chat_room_customer_null_by_default(self):
        room = make_chat_room(self.channel)
        self.assertIsNone(room.customer)

    def test_chat_room_customer_assignment(self):
        room = make_chat_room(self.channel, customer=self.customer)
        self.assertEqual(room.customer, self.customer)

    def test_chat_room_filter_by_customer_id(self):
        make_chat_room(self.channel, customer=self.customer)
        channel2 = make_channel(self.user.__class__.objects.create_user(username="other2", password="x"))
        ChatRoom.objects.create(channel=channel2, platform_chat_id="chat_002")
        qs = ChatRoom.objects.filter(customer_id=self.customer.pk)
        self.assertEqual(qs.count(), 1)

    def test_chat_room_customer_set_null_on_delete(self):
        room = make_chat_room(self.channel, customer=self.customer)
        self.customer.delete()
        room.refresh_from_db()
        self.assertIsNone(room.customer_id)


class SessionCustomerFKTests(TestCase):
    def setUp(self):
        self.user = make_user()
        self.customer = make_customer()
        self.channel = make_channel(self.user)
        self.chat_room = make_chat_room(self.channel)

    def test_session_customer_null_by_default(self):
        session = Session.objects.create(chat_room=self.chat_room)
        self.assertIsNone(session.customer)

    def test_session_customer_assignment(self):
        session = Session.objects.create(chat_room=self.chat_room, customer=self.customer)
        self.assertEqual(session.customer, self.customer)

    def test_session_filter_by_customer_id(self):
        Session.objects.create(chat_room=self.chat_room, customer=self.customer)
        Session.objects.create(chat_room=self.chat_room)
        qs = Session.objects.filter(customer_id=self.customer.pk)
        self.assertEqual(qs.count(), 1)

    def test_session_customer_set_null_on_delete(self):
        session = Session.objects.create(chat_room=self.chat_room, customer=self.customer)
        self.customer.delete()
        session.refresh_from_db()
        self.assertIsNone(session.customer_id)
