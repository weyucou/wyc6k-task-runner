import uuid

from django.db import IntegrityError
from django.test import TestCase

from accounts.models import Customer


class CustomerModelTests(TestCase):
    def _make_customer(self, **kwargs) -> Customer:
        defaults = {
            "name": "Acme Corp",
        }
        defaults.update(kwargs)
        return Customer.objects.create(**defaults)

    def test_create(self):
        customer = self._make_customer()
        self.assertIsNotNone(customer.pk)
        self.assertIsInstance(customer.pk, uuid.UUID)
        self.assertEqual(customer.name, "Acme Corp")

    def test_str(self):
        customer = self._make_customer()
        self.assertEqual(str(customer), f"Acme Corp({customer.id})")

    def test_s3_context_prefix_auto_calculated(self):
        customer = self._make_customer()
        self.assertEqual(customer.s3_context_prefix, f"customers/{customer.id}/")

    def test_github_org_optional(self):
        customer = self._make_customer()
        self.assertIsNone(customer.github_org)

    def test_github_org_unique(self):
        self._make_customer(github_org="acme")
        with self.assertRaises(IntegrityError):
            self._make_customer(name="Other Corp", github_org="acme")

    def test_is_active_default(self):
        customer = self._make_customer()
        self.assertTrue(customer.is_active)

    def test_timestamps(self):
        customer = self._make_customer()
        self.assertIsNotNone(customer.created_datetime)
        self.assertIsNotNone(customer.updated_datetime)
