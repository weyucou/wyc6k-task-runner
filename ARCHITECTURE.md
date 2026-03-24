# Architecture — marvin-manager

## Package Structure

The repository contains two implementations:

| Package | Status | Description |
|---------|--------|-------------|
| `mrvn/` | Legacy | Django-based implementation with ORM, pgvector memory, and REST API |
| `marvin/` | Active | Pure-Python, stateless SQS worker — no Django dependency |

`marvin/` replaces `mrvn/` for the stateless SQS-based runtime. `mrvn/` remains in the repository during transition.

### `marvin/` — Stateless SQS Worker

The package receives a hydrated `TaskEnvelope` from SQS, pulls an S3 context bundle via `ContextBundleService`, and runs an `AgentRunner` loop to completion. There is no database or ORM dependency.

Key files:

| File | Purpose |
|------|---------|
| `marvin/models.py` | `AgentConfig`, `TaskEnvelope`, `LLMProvider`, `ToolProfile` — Pydantic models |
| `marvin/worker.py` | SQS consumer loop entry point (`python -m marvin`) |
| `marvin/runner.py` | `AgentRunner` — orchestrates tool-call loop with rate limiting |
| `marvin/context.py` | `ContextBundleService` — reads customer context from S3 |
| `marvin/llm/` | LLM clients (Anthropic, Gemini, OpenAI, Ollama) |
| `marvin/tools/` | Tool base classes, registry, built-in tools, coding tools |
| `marvin/rate_limiter.py` | Thread-safe sliding-window rate limiter (keyed by agent name) |

#### TaskEnvelope flow

```
SQS → poll_once() → TaskEnvelope.model_validate()
    → ContextBundleService.pull(s3_prefix)  # fetch CLAUDE.md, SOPs, memories
    → AgentRunner(agent=AgentConfig, session_id=str)
    → runner.chat(user_message, ...)
    → LLM client (generate / generate_with_tools loop)
    → result dict → delete SQS message
```

#### MemorySearchTool in stateless mode

`MemorySearchTool.execute()` returns an empty result in stateless mode. The tool is still registered in the registry so agents that reference it do not error — they simply receive "Memory search not available in stateless mode."



marvin-manager is the **agent runtime** of the WYC6k system. It provides the LLM harness, tool profiles, and hybrid vector memory that agent workers use to process tasks.

For the full system architecture see [weyucou/wyc6k-spec](https://github.com/weyucou/wyc6k-spec).

## Responsibility

marvin-manager is a stateless worker. It receives a hydrated context bundle (pulled from S3 by the worker entrypoint) and a task description, then runs an `AgentRunner` loop to completion. It does not own scheduling, dispatch, or customer identity — those belong to jones.

## Django Apps

| App | Responsibility |
|-----|---------------|
| `accounts` | User accounts; `Customer` model with `customer_id` FK chain |
| `agents` | `Agent` model, `ToolRegistry`, `AgentRunner`, LLM clients |
| `memory` | `Session`, `EmbeddingChunk`, `ConversationSummary` — hybrid vector memory |

## AgentRunner

`mrvn/agents/runner.py` — orchestrates the tool-call loop.

- Up to 10 iterations per task
- Rate limiting between LLM calls
- Context compression via `ConversationSummary` when approaching token limits
- Accepts a `context_bundle: ProjectContextBundle` loaded from S3 at worker startup

## Tool System

All tools inherit from `BaseTool` (`mrvn/agents/tools/base.py`) and are registered via `register_builtin_tools()`.

### Tool Profiles

| Profile | Purpose |
|---------|---------|
| `MINIMAL` | No tools — LLM response only |
| `CODING` | File I/O, shell, web fetch/search, sub-agent sessions, image analysis, browser |
| `MESSAGING` | Messaging-channel tools |
| `FULL` | All registered tools |

Per-agent allow/deny lists can further restrict or extend a profile.

### Key Coding Tools

| Tool | `require_approval` | Description |
|------|-------------------|-------------|
| `ReadTool` | No | Read file contents |
| `WriteTool` | Yes | Write/overwrite a file |
| `EditTool` | Yes | Targeted string replacement |
| `ExecTool` | Yes | Run shell commands (includes `gh` CLI) |
| `SessionsSpawnTool` | Yes | Spawn a Claude CLI sub-agent |
| `SessionsSendTool` | Yes | Send prompt to a sub-agent |
| `BrowserTool` | Yes | Browser automation via Playwright |

## Memory System

Hybrid search over conversation history: pgvector HNSW index (semantic) + BM25 full-text (keyword), blended at configurable weights (default 0.7:0.3).

Partitioned by `(customer_id, agent_id)` — tenant isolation enforced at the query level.

`ConversationSummary` compresses old context when the token budget is exceeded, preserving recent messages verbatim.

## LLM Clients

`BaseLLMClient` ABC with implementations for:
- Anthropic Claude
- Google Gemini
- Ollama (local)
- OpenAI / vLLM

Provider is configured per `Agent` model instance.

## Multi-Tenancy

Customer isolation is enforced via `customer_id` FK on `Agent` and `EmbeddingChunk`. All memory queries are scoped to `(customer_id, agent_id)`. Cross-customer data access is not possible through the standard query paths.
