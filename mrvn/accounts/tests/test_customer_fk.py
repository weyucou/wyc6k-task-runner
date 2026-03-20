from django.contrib.auth import get_user_model
from django.test import TestCase

from accounts.models import Customer
from agents.models import Agent

User = get_user_model()


def make_customer(**kwargs) -> Customer:
    defaults = {"name": "Test Corp"}
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
