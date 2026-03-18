from django.test import TestCase

from accounts.models import Customer


class CustomerModelTest(TestCase):
    def test_create_customer(self):
        customer = Customer.objects.create(
            name="Acme Corp",
            github_org="acme",
            s3_context_prefix="s3://weyucou-agent-contexts/cust_acme/",
        )
        self.assertEqual(customer.name, "Acme Corp")
        self.assertEqual(customer.github_org, "acme")
        self.assertTrue(customer.is_active)

    def test_str(self):
        customer = Customer(name="Test Org")
        self.assertEqual(str(customer), "Test Org")

    def test_github_org_unique(self):
        Customer.objects.create(
            name="Org A",
            github_org="same-org",
            s3_context_prefix="s3://bucket/a/",
        )
        from django.db import IntegrityError

        with self.assertRaises(IntegrityError):
            Customer.objects.create(
                name="Org B",
                github_org="same-org",
                s3_context_prefix="s3://bucket/b/",
            )

    def test_is_active_default_true(self):
        customer = Customer.objects.create(
            name="Active Corp",
            github_org="active-corp",
            s3_context_prefix="s3://bucket/active/",
        )
        self.assertTrue(customer.is_active)

    def test_timestamps_set_on_create(self):
        customer = Customer.objects.create(
            name="Stamped Corp",
            github_org="stamped-corp",
            s3_context_prefix="s3://bucket/stamped/",
        )
        self.assertIsNotNone(customer.created_datetime)
        self.assertIsNotNone(customer.updated_datetime)
