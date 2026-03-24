# CLAUDE.md

This file provides guidance to AI agents (Claude Code and similar) when working with this repository. Keep it accurate and concise — agents read it on every session start.

Org-wide conventions (coding style, error handling, branching, testing, tool preferences) are defined in the system-level `~/.claude/CLAUDE.md` and apply automatically. Only sections below that are project-specific or override an org default are included here.

## Project Overview

marvin-manager is the **agent runtime** of the WYC6k multi-tenant agent harness. It is a Django 5.2 application that dequeues `TaskEnvelope` messages from SQS, runs an LLM agent in a tool-call loop against a GitHub issue, and writes memory back to S3 on completion. It does not communicate directly with human users — message intake and dispatch are handled by the upstream dispatcher layer (jones).

See [weyucou/wyc6k-spec](https://github.com/weyucou/wyc6k-spec) for full system architecture.

## Key Files

| File | Purpose |
|------|---------|
| `mrvn/mrvn/settings.py` | Django configuration; all env vars loaded via `os.getenv()` |
| `mrvn/mrvn/urls.py` | Root URL configuration |
| `mrvn/manage.py` | Django management entry point |
| `mrvn/agents/` | LLM agent configuration and tool registry |
| `mrvn/agents/tools/` | Tool implementations (`base.py`, `builtin.py`, `coding.py`) |
| `mrvn/memory/` | Session storage, vector embeddings, and hybrid memory search |
| `mrvn/autoreply/` | Response routing rules |
| `pyproject.toml` | Project metadata, dependencies, and tool configuration |

## Coding Conventions

- **Language:** Python 3.14
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

## Testing

- **Runner:** `pytest` (via `uv run poe test`)
- **Directory:** `tests/` within each Django app

## Do Not

- Do not add a new tool without inheriting from `BaseTool` and registering via `register_builtin_tools()`
- Do not add `channels`-app (Telegram/Slack) dependencies to core agent logic — see issue #37
