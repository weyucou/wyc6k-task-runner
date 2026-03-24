# CLAUDE.md

This file provides guidance to AI agents (Claude Code and similar) when working with this repository. Keep it accurate and concise — agents read it on every session start.

Org-wide conventions (coding style, error handling, branching, testing, tool preferences) are defined in the system-level `~/.claude/CLAUDE.md` and apply automatically. Only include sections below that are **project-specific** or **override an org default**.

## Project Overview

marvin-manager is the agent runtime of the WYC6k system. It provides the LLM harness, tool profiles, and SQS-based task dispatch that agent workers use to process tasks.

The repository contains two packages:

- `marvin_manager/` — the active, pure-Python, stateless SQS worker (no Django). This is the package under active development.
- `mrvn/` — the legacy Django implementation (ORM, pgvector memory, REST API). Kept during transition; do not delete.

For system-wide design, see [weyucou/wyc6k-spec](https://github.com/weyucou/wyc6k-spec).

## Key Files

| File | Purpose |
|------|---------|
| `marvin_manager/models.py` | `AgentConfig`, `TaskEnvelope`, `LLMProvider`, `ToolProfile` — Pydantic models replacing Django ORM |
| `marvin_manager/worker.py` | SQS consumer loop entry point (`python -m marvin_manager`) |
| `marvin_manager/runner.py` | `AgentRunner` — orchestrates the tool-call loop |
| `marvin_manager/context.py` | `ContextBundleService` — pulls customer context (CLAUDE.md, SOPs, memories) from S3 |
| `marvin_manager/llm/factory.py` | `create_llm_client()`, `create_client_from_agent_config()` |
| `marvin_manager/rate_limiter.py` | Thread-safe sliding-window rate limiter, keyed by agent name (str) |
| `marvin_manager/tools/builtin.py` | Core built-in tools; `MemorySearchTool` returns empty in stateless mode |
| `marvin_manager/tools/coding.py` | File I/O, shell, web fetch/search, browser, sub-agent session tools |
| `tests/` | pytest tests for `marvin_manager` (no Django test runner) |
| `pyproject.toml` | Project config; Django and related deps have been removed |

## Do Not

- Do not add Django, psycopg, or DRF dependencies to `marvin_manager/`.
- Do not delete `mrvn/` — it is still the legacy app.
- The `RateLimiterRegistry` in `marvin_manager/rate_limiter.py` uses `str` keys (agent names), not `int` (DB IDs) as in the legacy `mrvn/commons/rate_limiter.py`.
