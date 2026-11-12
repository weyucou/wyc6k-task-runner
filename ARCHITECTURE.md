# Architecture — marvin

marvin is the **agent runtime** of the WYC6k system. It provides the LLM harness, tool profiles, and SQS-based task dispatch that agent workers use to process tasks.

For the full system architecture see [weyucou/wyc6k-spec](https://github.com/weyucou/wyc6k-spec).

## Responsibility

marvin is a stateless worker. It receives a hydrated context bundle (pulled from S3 by the worker entrypoint) and a task description, then runs an `AgentRunner` loop to completion. It does not own scheduling, dispatch, or customer identity — those belong to jones.

## Worker Paths

Three execution modes share the `marvin/` codebase. Choose based on deployment context:

| Mode | Entry point | Invocation | When to use |
|------|------------|------------|-------------|
| **SQS consumer** | `marvin/worker.py` | `python -m marvin` | Long-lived container polling an SQS queue (e.g. ECS service) |
| **One-shot Fargate** | `entrypoint.py` | container `ENTRYPOINT` | ECS Fargate RunTask per task — container starts, runs one `TaskEnvelope`, exits |
| **Launcher** | `marvin/launchers/` | `python -m marvin.launchers` | Polls SQS and starts one one-shot container per `TaskEnvelope`: local `docker run` (`COMPUTE_BACKEND=docker`) or `ecs:RunTask` (`COMPUTE_BACKEND=ecs`) |

### One-shot Fargate flow (`entrypoint.py`)

```
ECS RunTask (TASK_ENVELOPE_JSON env var)
    ↓
entrypoint.py
    1. Parse   — TaskEnvelope.model_validate_json(TASK_ENVELOPE_JSON)
    2. Resolve — CredentialResolver.resolve(envelope) → GITHUB_TOKEN / api_key
    3. Pull    — ContextBundleService.pull(s3_context_prefix) → CustomerContextBundle
    4. Run     — AgentRunner.chat(user_message, ...)
    5. Memory  — ContextBundleService.push_memory(...) → daily memory file on S3
    6. Exit    — 0 (success) | 1 (task failure) | 2 (setup/config failure)
```

The one-shot path never polls SQS. `TASK_ENVELOPE_JSON` is injected by the ECS task definition override at launch time (see `wyc6k-infra`).

### Launcher flow (`marvin/launchers/`)

The launcher runs the one-shot path under the production contract (same image, same `entrypoint.py`, `TASK_ENVELOPE_JSON` env var) from an SQS queue. Usage and settings: [README → Local Docker run](README.md#local-docker-run).

```
SQS receive_message (consumer.poll_once)
    ↓ TaskEnvelope.model_validate_json(body)      invalid → message left (→ DLQ)
get_launcher()  ← COMPUTE_BACKEND
    ├─ docker: docker run --rm -e TASK_ENVELOPE_JSON -e CUSTOMER_ID … <AGENT_WORKER_IMAGE>   (blocks, heartbeats)
    └─ ecs:    ecs.run_task(FARGATE, awsvpc, containerOverrides[agent-worker]: TASK_ENVELOPE_JSON, CUSTOMER_ID)
    ↓
entrypoint.py in the container (exit 0 | 1 | 2)
    ↓
LaunchResult.success → delete_message, else leave the message for redelivery
```

The body is forwarded unchanged as `TASK_ENVELOPE_JSON`; it is validated, never re-serialised.

| Backend | `success` means | On failure |
|---------|-----------------|------------|
| `docker` | the container exited `0` | exit `1`/`2` (or a docker error such as `125`) leaves the message: it is redelivered after `VISIBILITY_TIMEOUT`, then dead-lettered by the queue's redrive policy |
| `ecs` | ECS accepted the task (task ARN, no `failures`) | an API error or `failures` leaves the message. `run_task` is asynchronous, so failures after dispatch surface in ECS task state / CloudWatch, not SQS |

- **Heartbeat**: while a container runs, the docker backend extends the message visibility every `VISIBILITY_TIMEOUT/2` (`consumer.extend_visibility`), so a long task is not launched twice. SQS caps visibility at 12 h after receipt.
- **Sequential**: envelopes run one at a time; only the running message is heartbeated, so keep `MAX_MESSAGES=1` with `docker` (the launcher logs a warning at startup otherwise).
- **At-least-once**: if `delete_message` fails after a successful task, the launcher logs `Task succeeded but delete_message failed` and the error propagates to the poll loop's retry. The message is redelivered, so the task runs again.
- **No secrets in argv**: values reach the container via the `docker` process env and name-only `-e NAME`, so neither the envelope nor credentials appear in `ps`. Linux caps one env var at 128 KiB; a larger envelope fails to start and its message is left.
- **ECS override limit**: container overrides total at most 8 KiB, so a large `TASK_ENVELOPE_JSON` (e.g. long `conversation_history`) fails `run_task` with `InvalidParameterException`. This limit already applies to the production path.
- **ECS config**: the infra stack (`wyc6k-infra` `stacks/ecs-agent-worker.yaml`) publishes the cluster and task-definition ARNs to SSM at `/wyc6k/agent-infra/<stage>/agent_worker_cluster_arn` and `.../agent_worker_task_definition_arn`; the container is named `agent-worker`.
- **Shutdown**: SIGTERM or SIGINT sent to the launcher process stops the loop once the running container exits; the launcher never stops a container itself.
  - Ctrl-C in a terminal reaches the whole foreground process group, including the `docker run` child, which forwards SIGINT to the container. The agent is interrupted and exits non-zero, and its message is left for redelivery.
  - `docker compose stop` sends SIGTERM, then kills the launcher after the 10 s grace period. A running agent container keeps going (`--rm` removes it), but nothing deletes its message, so it is redelivered once visibility expires and the task runs again.
- **Docker socket**: the `local-runner` compose profile mounts `/var/run/docker.sock`, which gives the launcher full control of the host Docker daemon. Local use only.
- **Router Lambda**: the planned `duration_hint` router can reuse `get_launcher("ecs")` / `EcsLauncher` directly.

### Building & publishing the agent-worker image

The image can be built and pushed to ECR either automatically (via `.github/workflows/build-image.yml` on push to `main`) or manually from a developer workstation. Both paths produce the same artifact tags: `latest` plus `sha-<short-sha>`.

Prerequisites:

- Docker with BuildKit + buildx (`docker buildx version` to confirm)
- AWS credentials authenticated to the **dev** account with permission to push to the `weyucou/agent-worker` ECR repository (the active shell is the auth boundary — do **not** pass `--profile` flags or set `AWS_PROFILE` in these snippets unless your local setup requires it)
- The ECR repository `weyucou/agent-worker` must exist in the target account/region (create it once via `aws ecr create-repository --repository-name weyucou/agent-worker`)

Manual build + push:

```bash
# 0. Substitute these for your environment
AWS_ACCOUNT_ID=610714125210               # weyucou dev account
AWS_REGION=us-west-2                      # ECR repo region
SHORT_SHA=$(git rev-parse --short HEAD)
ECR_URI=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/weyucou/agent-worker

# 1. Build (multi-stage, fargate dep-group, ~349 MB)
docker buildx build --platform linux/amd64 \
  -t weyucou/agent-worker:local \
  -t ${ECR_URI}:latest \
  -t ${ECR_URI}:sha-${SHORT_SHA} \
  --load .

# 2. Verify size before pushing (AC #7: < 2 GB)
docker image inspect weyucou/agent-worker:local --format '{{.Size}}'

# 3. Authenticate Docker to ECR (12-hour token)
aws ecr get-login-password --region ${AWS_REGION} \
  | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# 4. Push both tags
docker push ${ECR_URI}:latest
docker push ${ECR_URI}:sha-${SHORT_SHA}

# 5. Verify
aws ecr describe-images --repository-name weyucou/agent-worker --region ${AWS_REGION} \
  --query 'imageDetails[?contains(imageTags, `latest`) || contains(imageTags, `sha-'${SHORT_SHA}'`)].[imageTags,imageSizeInBytes,imagePushedAt]' \
  --output table
```

Automated build + push (GHA workflow):

The workflow at `.github/workflows/build-image.yml` performs the same five steps on push to `main` using the OIDC IAM role configured via repository variables `AWS_REGION` and `ECR_PUSH_ROLE_ARN`. Until those repo variables are set the workflow will fail at step 3 (`configure-aws-credentials`) — until then, use the manual procedure above.

## Package Structure

| File | Purpose |
|------|---------|
| `marvin/models.py` | `AgentConfig`, `TaskEnvelope`, `LLMProvider`, `ToolProfile` — Pydantic models |
| `marvin/worker.py` | SQS consumer loop entry point (`python -m marvin`) |
| `marvin/launchers/` | Launcher (`python -m marvin.launchers`): SQS → one agent-worker container per envelope, `COMPUTE_BACKEND=docker\|ecs` |
| `marvin/runner.py` | `AgentRunner` — orchestrates tool-call loop with rate limiting |
| `marvin/context.py` | `ContextBundleService` — reads customer context from S3 |
| `marvin/llm/` | LLM clients (Anthropic, Gemini, OpenAI, Ollama) |
| `marvin/tools/` | Tool base classes, registry, built-in tools, coding tools |
| `marvin/rate_limiter.py` | Thread-safe sliding-window rate limiter (keyed by agent name) |

## TaskEnvelope flow

```
SQS → poll_once() → TaskEnvelope.model_validate()
    → ContextBundleService.pull(s3_prefix)  # fetch CLAUDE.md, SOPs, memories
    → AgentRunner(agent=AgentConfig, session_id=str)
    → runner.chat(user_message, ...)
    → LLM client (generate / generate_with_tools loop)
    → result dict → delete SQS message
```

## MemorySearchTool in stateless mode

`MemorySearchTool.execute()` returns an empty result in stateless mode. The tool is still registered in the registry so agents that reference it do not error — they simply receive "Memory search not available in stateless mode."

## AgentRunner

`marvin/runner.py` — orchestrates the tool-call loop.

- Up to 10 iterations per task
- Rate limiting between LLM calls
- Accepts a `context_bundle: ProjectContextBundle` loaded from S3 at worker startup

## Tool System

All tools inherit from `BaseTool` (`marvin/tools/base.py`) and are registered via `register_builtin_tools()`.

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

## LLM Clients

`BaseLLMClient` ABC with implementations for:
- Anthropic Claude
- Google Gemini
- Ollama (local)
- OpenAI / vLLM

Provider is configured per `AgentConfig` instance.

## Multi-Tenancy

Customer isolation is enforced via `customer_id` on `AgentConfig`. All operations are scoped to the customer context passed in the `TaskEnvelope`.
