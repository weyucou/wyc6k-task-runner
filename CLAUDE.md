# CLAUDE.md

This file provides guidance to AI agents (Claude Code and similar) when working with this repository. Keep it accurate and concise — agents read it on every session start.

## Project Overview

marvin-manager is the agent runtime of the **WYC6k** multi-tenant agent harness. It is a Django 5.2 application that manages AI agent configuration, handles multi-channel messaging (Telegram, Slack), maintains conversation memory, and routes messages to multiple LLM backends (Anthropic Claude, Google Gemini, Ollama, vLLM). It exposes a Django Admin interface and a set of management commands for channel setup.

See [weyucou/wyc6k-spec](https://github.com/weyucou/wyc6k-spec) for full system architecture.

## Key Files

| File | Purpose |
|------|---------|
| `mrvn/mrvn/settings.py` | Django configuration; all env vars loaded via `os.getenv()` |
| `mrvn/mrvn/urls.py` | Root URL configuration |
| `mrvn/manage.py` | Django management entry point |
| `mrvn/agents/` | LLM agent configuration and tool registry |
| `mrvn/agents/tools/` | Tool implementations (`base.py`, `builtin.py`, `coding.py`) |
| `mrvn/channels/` | Messaging channel integrations (Telegram, Slack) |
| `mrvn/memory/` | Conversation storage and session management |
| `mrvn/autoreply/` | Response routing rules |
| `pyproject.toml` | Project metadata, dependencies, and tool configuration |

## Coding Conventions

- **Language:** Python 3.14
- **Formatter:** `ruff format` — run before every commit
- **Linter:** `ruff check` (via `uv run poe check`)
- **Type checker:** `pyright` (via `uv run pyright`)
- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes and Django models
- **Data models:** Django ORM models for persistence; `pydantic.BaseModel` for serialization schemas
- **Tools:** All agent tools inherit from `BaseTool` (`mrvn/agents/tools/base.py`) and are registered via `register_builtin_tools()`

## Development Commands

| Command | Purpose |
|---------|---------|
| `uv sync` | Install all dependencies |
| `uv run poe check` | Run linters (ruff) |
| `uv run pyright` | Run type checker |
| `uv run poe test` | Run tests |
| `uv run python manage.py runserver` | Start dev server (run from `mrvn/`) |
| `uv run python manage.py onboard` | Interactive setup wizard |
| `uv run python manage.py run_telegram` | Run Telegram bot |
| `uv run python manage.py run_slack` | Run Slack bot |

## Testing

- **Runner:** `pytest` (via `uv run poe test`)
- **Directory:** `tests/` within each Django app
- **Coverage:** All new logic must have unit tests. Bug fixes require a regression test that fails without the fix.
- **TDD enforcement** — follow RED-GREEN-REFACTOR strictly:
  1. Write a failing test (RED)
  2. Write the minimum code to pass (GREEN)
  3. Refactor while keeping tests green (REFACTOR)

  Do not write implementation code before a failing test exists. Do not mark a task complete until all tests pass.

## Error Handling

- Always capture exception instances: `except Exception as exc:` — never pass the class
- Never use `str(Exception)` in error handlers; use `str(exc)` or `repr(exc)`
- Use the `logging` module (not `print()`); include context with each logged exception
- Do not swallow exceptions silently — at minimum log a warning with the exception

## Branching

All branches follow `<type>/<issue-number>-<short-description>`:

| Prefix | Purpose |
|--------|---------|
| `feature/` | New functionality |
| `fix/` | Bug resolution |
| `hotfix/` | Urgent production fix |
| `chore/` | Maintenance (deps, docs, config) |
| `release/` | Release preparation |

Examples: `feature/42-add-gemini-provider`, `fix/108-telegram-reconnect`, `chore/63-upgrade-django`

## Do Not

- Do not use `os.environ[]` at module level in `settings.py` — use `os.getenv()` with a default or load inside a function (bare `os.environ[]` crashes test collection)
- Do not use `print()` for application logging — use the `logging` module
- Do not commit secrets, API keys (`ANTHROPIC_API_KEY`, `BRAVE_SEARCH_API_KEY`), or `.env` files to the repository
- Do not skip pre-commit hooks (`--no-verify`) without explicit user approval
- Do not mark a task complete until all tests pass
- Do not add a new tool without inheriting from `BaseTool` and registering via `register_builtin_tools()`
