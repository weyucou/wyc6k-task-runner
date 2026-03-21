import getpass
import logging
import os
import sys
from typing import TYPE_CHECKING

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError
from django.db import DatabaseError, connection

if TYPE_CHECKING:
    from argparse import ArgumentParser

logger = logging.getLogger(__name__)

User = get_user_model()

MIN_PASSWORD_LENGTH = 8

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███╗   ███╗ █████╗ ██████╗ ██╗   ██╗██╗███╗   ██╗          ║
║   ████╗ ████║██╔══██╗██╔══██╗██║   ██║██║████╗  ██║          ║
║   ██╔████╔██║███████║██████╔╝██║   ██║██║██╔██╗ ██║          ║
║   ██║╚██╔╝██║██╔══██║██╔══██╗╚██╗ ██╔╝██║██║╚██╗██║          ║
║   ██║ ╚═╝ ██║██║  ██║██║  ██║ ╚████╔╝ ██║██║ ╚████║          ║
║   ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚═╝  ╚═══╝          ║
║                                                              ║
║                    Welcome to Marvin                         ║
║           AI Assistant Platform Setup Wizard                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


class Command(BaseCommand):
    help = "Interactive setup wizard for Marvin"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--skip-migrations",
            action="store_true",
            help="Skip running database migrations",
        )
        parser.add_argument(
            "--skip-superuser",
            action="store_true",
            help="Skip superuser creation",
        )
        parser.add_argument(
            "--non-interactive",
            action="store_true",
            help="Run without prompts (uses defaults or environment variables)",
        )

    def handle(self, *args, **options) -> None:
        self.non_interactive = options["non_interactive"]

        self.stdout.write(self.style.SUCCESS(BANNER))

        # Step 1: Check system requirements
        self._check_requirements()

        # Step 2: Check database connection
        self._check_database()

        # Step 3: Run migrations
        if not options["skip_migrations"]:
            self._run_migrations()

        # Step 4: Create superuser
        if not options["skip_superuser"]:
            self._create_superuser()

        # Step 5: Create default agent
        self._create_default_agent()

        # Step 6: Display completion and next steps
        self._display_completion()

    def _check_requirements(self) -> None:
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("Step 1: Checking System Requirements"))
        self.stdout.write("=" * 60 + "\n")

        # Check Python version
        py_version = sys.version_info
        self.stdout.write(f"  Python: {py_version.major}.{py_version.minor}.{py_version.micro}")
        if py_version < (3, 14):
            self.stdout.write(self.style.WARNING("    Warning: Python 3.14+ recommended"))
        else:
            self.stdout.write(self.style.SUCCESS("    OK"))

        # Check required packages
        packages = ["django", "anthropic", "slack_bolt", "telegram"]
        for pkg in packages:
            try:
                __import__(pkg)
                self.stdout.write(f"  {pkg}: " + self.style.SUCCESS("installed"))
            except ImportError:
                self.stdout.write(f"  {pkg}: " + self.style.WARNING("not found"))

        self.stdout.write("")

    def _check_database(self) -> None:
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("Step 2: Checking Database Connection"))
        self.stdout.write("=" * 60 + "\n")

        db_settings = settings.DATABASES["default"]
        self.stdout.write(f"  Engine: {db_settings['ENGINE']}")
        self.stdout.write(f"  Host: {db_settings.get('HOST', 'localhost')}")
        self.stdout.write(f"  Database: {db_settings.get('NAME', 'default')}")

        try:
            connection.ensure_connection()
            self.stdout.write(self.style.SUCCESS("  Connection: OK\n"))
        except DatabaseError as e:
            self.stdout.write(self.style.ERROR(f"  Connection: FAILED - {e}\n"))
            if not self.non_interactive:
                if not self._confirm("Database connection failed. Continue anyway?"):
                    raise CommandError("Setup cancelled - fix database connection and retry.") from None
            else:
                raise CommandError(f"Database connection failed: {e}") from e

    def _run_migrations(self) -> None:
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("Step 3: Running Database Migrations"))
        self.stdout.write("=" * 60 + "\n")

        try:
            call_command("migrate", verbosity=1)
            self.stdout.write(self.style.SUCCESS("\n  Migrations completed successfully\n"))
        except CommandError as e:
            self.stdout.write(self.style.ERROR(f"\n  Migration failed: {e}\n"))
            raise CommandError(f"Migration failed: {e}") from e

    def _create_superuser(self) -> None:
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("Step 4: Create Admin User"))
        self.stdout.write("=" * 60 + "\n")

        # Check if any superuser exists
        if User.objects.filter(is_superuser=True).exists():
            existing = User.objects.filter(is_superuser=True).first()
            self.stdout.write(f"  Superuser already exists: {existing.username}")  # type: ignore[union-attr]
            if not self.non_interactive:
                if not self._confirm("  Create another superuser?"):
                    self.stdout.write("  Skipping superuser creation\n")
                    return
            else:
                self.stdout.write("  Skipping superuser creation\n")
                return

        if self.non_interactive:
            # Use environment variables
            username = os.getenv("DJANGO_SUPERUSER_USERNAME", "admin")
            email = os.getenv("DJANGO_SUPERUSER_EMAIL", "admin@localhost")
            password = os.getenv("DJANGO_SUPERUSER_PASSWORD")

            if not password:
                raise CommandError("DJANGO_SUPERUSER_PASSWORD required in non-interactive mode")

            User.objects.create_superuser(username=username, email=email, password=password)
            self.stdout.write(self.style.SUCCESS(f"  Created superuser: {username}\n"))
        else:
            self._create_superuser_interactive()

    def _create_superuser_interactive(self) -> None:
        username = self._prompt("  Username", default="admin")
        email = self._prompt("  Email", default=f"{username}@localhost")

        while True:
            password = self._prompt_secret("  Password")
            password_confirm = self._prompt_secret("  Confirm password")

            if password != password_confirm:
                self.stdout.write(self.style.ERROR("  Passwords do not match. Try again."))
                continue

            if len(password) < MIN_PASSWORD_LENGTH:
                self.stdout.write(self.style.WARNING(f"  Warning: Password is less than {MIN_PASSWORD_LENGTH} chars"))
                if not self._confirm("  Use this password anyway?"):
                    continue
            break

        User.objects.create_superuser(username=username, email=email, password=password)
        self.stdout.write(self.style.SUCCESS(f"\n  Created superuser: {username}\n"))

    def _create_default_agent(self) -> None:
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("Step 5: Configure Default Agent"))
        self.stdout.write("=" * 60 + "\n")

        from agents.models import Agent, LLMProvider  # noqa: PLC0415

        # Check if default agent exists
        if Agent.objects.filter(name="Default Assistant").exists():
            self.stdout.write("  Default agent already exists\n")
            return

        if self.non_interactive:
            provider = os.getenv("DEFAULT_LLM_PROVIDER", LLMProvider.ANTHROPIC.value)
            model = os.getenv("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
        else:
            self.stdout.write("  Select LLM Provider:")
            self.stdout.write("    1. Anthropic (Claude) - recommended")
            self.stdout.write("    2. Google (Gemini)")
            self.stdout.write("    3. Ollama (Local)")
            self.stdout.write("    4. vLLM (Local)")

            choice = self._prompt("  Choice [1-4]", default="1")

            provider_map = {
                "1": (LLMProvider.ANTHROPIC.value, "claude-sonnet-4-20250514"),
                "2": (LLMProvider.GEMINI.value, "gemini-2.0-flash"),
                "3": (LLMProvider.OLLAMA.value, "llama3.2"),
                "4": (LLMProvider.VLLM.value, "meta-llama/Llama-3.2-8B"),
            }

            provider, model = provider_map.get(choice, provider_map["1"])

        # Get base_url for local providers
        base_url = ""
        if provider in (LLMProvider.OLLAMA.value, LLMProvider.VLLM.value):
            default_urls = {
                LLMProvider.OLLAMA.value: "http://localhost:11434",
                LLMProvider.VLLM.value: "http://localhost:8000/v1",
            }
            if self.non_interactive:
                base_url = os.getenv("DEFAULT_LLM_BASE_URL", default_urls[provider])
            else:
                base_url = self._prompt("  Base URL", default=default_urls[provider])

        Agent.objects.create(
            name="Default Assistant",
            description="Default AI assistant for Marvin",
            provider=provider,
            model_name=model,
            base_url=base_url,
            system_prompt="You are a helpful AI assistant.",
            is_active=True,
            rate_limit_enabled=provider in (LLMProvider.ANTHROPIC.value, LLMProvider.GEMINI.value),
            rate_limit_rpm=60 if provider in (LLMProvider.ANTHROPIC.value, LLMProvider.GEMINI.value) else 0,
        )

        self.stdout.write(self.style.SUCCESS(f"\n  Created default agent with {provider} ({model})\n"))

    def _display_completion(self) -> None:
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("Setup Complete!"))
        self.stdout.write("=" * 60 + "\n")

        self.stdout.write(
            self.style.SUCCESS("""
Marvin has been configured successfully!

Next Steps:
-----------

1. Start the development server:
   uv run python manage.py runserver

2. Access the admin panel:
   http://127.0.0.1:8000/admin/

3. Configure messaging channels:

   Telegram:
   ---------
   uv run python manage.py setup_telegram --owner <username>

   Slack:
   ------
   uv run python manage.py setup_slack --owner <username>

4. Add API credentials for your LLM provider in the admin panel:
   Admin -> Agents -> Agent Credentials

5. (Optional) Configure rate limiting per agent in the admin panel

Documentation:
--------------
See README.md for detailed configuration options.

""")
        )

    def _prompt(self, label: str, default: str | None = None) -> str:
        prompt_text = f"{label} [{default}]: " if default else f"{label}: "
        self.stdout.write(prompt_text, ending="")
        value = input().strip()
        if not value and default:
            return default
        if not value:
            raise CommandError(f"{label} is required.")
        return value

    def _prompt_secret(self, label: str) -> str:
        try:
            return getpass.getpass(f"{label}: ")
        except EOFError:
            raise CommandError(f"{label} is required.") from None

    def _confirm(self, message: str) -> bool:
        self.stdout.write(f"{message} [y/N]: ", ending="")
        return input().strip().lower() in ("y", "yes")
