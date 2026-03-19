from django.db import IntegrityError
from django.test import TestCase

from accounts.models import Customer


class CustomerModelTests(TestCase):
    def _make_customer(self, **kwargs) -> Customer:
        defaults = {
            "name": "Acme Corp",
            "github_org": "acme",
            "s3_context_prefix": "cust_acme/",
        }
        defaults.update(kwargs)
        return Customer.objects.create(**defaults)

    def test_create(self):
        customer = self._make_customer()
        self.assertIsNotNone(customer.pk)
        self.assertEqual(customer.name, "Acme Corp")
        self.assertEqual(customer.github_org, "acme")
        self.assertEqual(customer.s3_context_prefix, "cust_acme/")

    def test_str(self):
        customer = self._make_customer()
        self.assertEqual(str(customer), "Acme Corp")

    def test_github_org_unique(self):
        self._make_customer()
        with self.assertRaises(IntegrityError):
            self._make_customer(name="Other Corp")

    def test_is_active_default(self):
        customer = self._make_customer()
        self.assertTrue(customer.is_active)

    def test_timestamps(self):
        customer = self._make_customer()
        self.assertIsNotNone(customer.created_datetime)
        self.assertIsNotNone(customer.updated_datetime)
