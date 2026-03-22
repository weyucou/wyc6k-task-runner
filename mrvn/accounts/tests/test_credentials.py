from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError
from django.db import IntegrityError
from django.test import TestCase

from accounts.models import CredentialService, Customer, CustomerCredential
from accounts.services import CustomerCredentialService


def _make_customer(**kwargs) -> Customer:
    defaults = {"name": "Acme Corp"}
    defaults.update(kwargs)
    return Customer.objects.create(**defaults)


def _make_credential(
    customer: Customer,
    service: CredentialService = CredentialService.github,
    arn: str = "arn:aws:secretsmanager:ap-northeast-1:123456789012:secret:acme-github",
) -> CustomerCredential:
    return CustomerCredential.objects.create(customer=customer, service=service, secret_arn=arn)


class CustomerCredentialModelTests(TestCase):
    def test_create(self):
        customer = _make_customer()
        cred = _make_credential(customer)
        self.assertEqual(cred.service, CredentialService.github)
        self.assertTrue(cred.secret_arn.startswith("arn:aws:"))

    def test_str(self):
        customer = _make_customer()
        cred = _make_credential(customer)
        self.assertIn("github", str(cred))

    def test_unique_together(self):
        customer = _make_customer()
        _make_credential(customer, CredentialService.github)
        with self.assertRaises(IntegrityError):
            _make_credential(customer, CredentialService.github)

    def test_different_services_allowed(self):
        customer = _make_customer()
        _make_credential(customer, CredentialService.github, "arn:aws:secretsmanager:::secret:gh")
        cred2 = _make_credential(customer, CredentialService.anthropic, "arn:aws:secretsmanager:::secret:ant")
        self.assertEqual(cred2.service, CredentialService.anthropic)

    def test_timestamps(self):
        customer = _make_customer()
        cred = _make_credential(customer)
        self.assertIsNotNone(cred.created_datetime)
        self.assertIsNotNone(cred.updated_datetime)


class CustomerCredentialServiceGetTests(TestCase):
    def setUp(self):
        self.customer = _make_customer()
        self.arn = "arn:aws:secretsmanager:ap-northeast-1:123:secret:gh-token"
        _make_credential(self.customer, CredentialService.github, self.arn)
        self.svc = CustomerCredentialService()

    @patch("accounts.services.boto3.client")
    def test_get_returns_secret_string(self, mock_boto_client: MagicMock):
        mock_sm = MagicMock()
        mock_sm.get_secret_value.return_value = {"SecretString": "ghp_token123"}
        mock_boto_client.return_value = mock_sm

        result = self.svc.get(self.customer.pk, CredentialService.github)

        mock_boto_client.assert_called_once_with("secretsmanager")
        mock_sm.get_secret_value.assert_called_once_with(SecretId=self.arn)
        self.assertEqual(result, "ghp_token123")

    @patch("accounts.services.boto3.client")
    def test_get_raises_if_no_credential(self, mock_boto_client: MagicMock):
        with self.assertRaises(CustomerCredential.DoesNotExist):
            self.svc.get(self.customer.pk, CredentialService.anthropic)
        mock_boto_client.assert_not_called()


class CustomerCredentialServiceInjectEnvTests(TestCase):
    def setUp(self):
        self.customer = _make_customer()
        self.svc = CustomerCredentialService()

    @patch("accounts.services.boto3.client")
    def test_inject_env_all_services(self, mock_boto_client: MagicMock):
        secrets = {
            "arn:gh": "ghp_token",
            "arn:ant": "sk-ant-token",
            "arn:gem": "AIza-token",
        }
        CustomerCredential.objects.create(  # noqa: S106
            customer=self.customer,
            service=CredentialService.github,
            secret_arn="arn:gh",  # noqa: S106
        )
        CustomerCredential.objects.create(
            customer=self.customer,
            service=CredentialService.anthropic,
            secret_arn="arn:ant",  # noqa: S106
        )
        CustomerCredential.objects.create(
            customer=self.customer,
            service=CredentialService.gemini,
            secret_arn="arn:gem",  # noqa: S106
        )

        mock_sm = MagicMock()
        mock_sm.get_secret_value.side_effect = lambda **kwargs: {"SecretString": secrets[kwargs["SecretId"]]}
        mock_boto_client.return_value = mock_sm

        env = self.svc.inject_env(self.customer.pk)

        self.assertEqual(env["GITHUB_TOKEN"], "ghp_token")
        self.assertEqual(env["ANTHROPIC_API_KEY"], "sk-ant-token")
        self.assertEqual(env["GEMINI_API_KEY"], "AIza-token")

    @patch("accounts.services.boto3.client")
    def test_inject_env_empty_when_no_credentials(self, mock_boto_client: MagicMock):
        env = self.svc.inject_env(self.customer.pk)
        self.assertEqual(env, {})
        mock_boto_client.assert_not_called()

    @patch("accounts.services.boto3.client")
    def test_inject_env_skips_failed_secret(self, mock_boto_client: MagicMock):
        CustomerCredential.objects.create(
            customer=self.customer,
            service=CredentialService.github,
            secret_arn="arn:gh",  # noqa: S106
        )
        mock_sm = MagicMock()
        mock_sm.get_secret_value.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}}, "GetSecretValue"
        )
        mock_boto_client.return_value = mock_sm

        env = self.svc.inject_env(self.customer.pk)

        self.assertEqual(env, {})
