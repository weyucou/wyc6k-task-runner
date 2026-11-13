# wyc6k-task-runner

The **agent runtime** of the WYC6k multi-tenant agent harness. Dequeues `TaskEnvelope` messages from SQS, runs an LLM agent in a tool-call loop against a GitHub issue, and writes memory back to S3 on completion.

## System Architecture

wyc6k-task-runner is the agent runtime of the **WYC6k** multi-tenant agent harness. See [weyucou/wyc6k-spec](https://github.com/weyucou/wyc6k-spec) for the full system architecture, and [ARCHITECTURE.md](ARCHITECTURE.md) for wyc6k-task-runner internals.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/weyucou/wyc6k-task-runner.git
cd wyc6k-task-runner
uv sync

# 2. Start LocalStack (SQS/S3)
docker compose up -d

# 3. Set required environment variables
export SQS_QUEUE_URL=http://localhost:4566/000000000000/tasks
export AWS_DEFAULT_REGION=us-east-1
export ANTHROPIC_API_KEY=your-key-here

# 4. Start the worker
python -m marvin
```

## Local Docker run

Runs real tasks in the production agent-worker image and `entrypoint.py`, launched with `TASK_ENVELOPE_JSON` exactly as ECS does. The launcher (`python -m marvin.launchers`) polls SQS and runs `docker run --rm -e TASK_ENVELOPE_JSON ... weyucou/agent-worker:local` per envelope, the local equivalent of `ecs:RunTask`. It deletes the SQS message only when the container exits `0`; exit `1`/`2` leaves it for redelivery (or the DLQ).

### Setup and run on Ubuntu Linux

Only step 1 is Ubuntu-specific; steps 2–5 work on any OS with Docker Engine, the compose plugin and AWS CLI v2. Steps 2–5 were tested end to end on Ubuntu 24.04 LTS; step 1 uses the commands from the official Docker, uv and AWS CLI install guides.

**1. Install the prerequisites** (once per machine)

```bash
# Docker Engine + compose plugin, from Docker's apt repository
sudo apt update
sudo apt install -y ca-certificates curl git unzip
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Run docker without sudo. Group changes apply to new logins: log out and back in,
# or use `newgrp` to start a shell that has the group now
sudo usermod -aG docker $USER
newgrp docker
docker run --rm hello-world

# uv (tests, and the launcher on the host); open a new shell afterwards so uv is on PATH
curl -LsSf https://astral.sh/uv/install.sh | sh

# AWS CLI v2 (x86_64 or aarch64)
curl "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o awscliv2.zip
unzip -q awscliv2.zip
sudo ./aws/install
aws --version
```

Docker from Ubuntu's own packages (`docker.io`, `docker-compose-v2`, `docker-buildx`) also works: skip the Docker block, but still join the `docker` group.

**2. Clone and build the images**

```bash
git clone https://github.com/weyucou/wyc6k-task-runner.git
cd wyc6k-task-runner
docker compose --profile local-runner build  # weyucou/agent-worker:local and the launcher image
```

**3. Start LocalStack and the launcher**

```bash
export ANTHROPIC_API_KEY=your-key-here  # forwarded to the agent container by name, never in argv
export GITHUB_TOKEN=your-token  # only needed for GitHub tools
docker compose --profile local-runner up -d  # plain `docker compose up -d` starts only LocalStack
until curl -fs http://localhost:4566/_localstack/health > /dev/null; do sleep 1; done  # wait for LocalStack
```

Until step 4 creates the queue, the launcher logs `SQS call failed; retrying` every `POLL_INTERVAL_SECONDS`. If `up` fails with `port is already allocated`, another container (often another project's LocalStack) holds port 4566; stop it first.

**4. Run a test task**

```bash
# Create the queue and the customer context in LocalStack
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1
aws --endpoint-url http://localhost:4566 sqs create-queue --queue-name tasks
aws --endpoint-url http://localhost:4566 s3 mb s3://local-context
echo "# Local customer" | aws --endpoint-url http://localhost:4566 s3 cp - s3://local-context/customers/cust-001/CLAUDE.md

# Enqueue a test envelope
cat > envelope.json <<'EOF'
{
  "task_id": "local-001",
  "customer_id": "cust-001",
  "session_id": "local",
  "agent": {"name": "local-agent", "tool_profile": "minimal"},
  "s3_context_prefix": "s3://local-context/customers/cust-001/projects/demo",
  "user_message": "Reply with one short sentence.",
  "enable_tools": false
}
EOF
aws --endpoint-url http://localhost:4566 sqs send-message \
  --queue-url http://localhost:4566/000000000000/tasks --message-body file://envelope.json

# Follow the launcher (agent logs are interleaved) until it logs "Task launched; message deleted".
# Ctrl-C here stops following the logs; the launcher keeps running.
docker compose logs -f launcher

# The agent's reply is in today's (UTC) memory entry
aws --endpoint-url http://localhost:4566 s3 ls --recursive s3://local-context/customers/cust-001/projects/demo/memory/
aws --endpoint-url http://localhost:4566 s3 cp \
  "s3://local-context/customers/cust-001/projects/demo/memory/$(date -u +%Y/%Y-%m-%d).md" -
```

**5. Stop everything**

```bash
docker compose --profile local-runner down
```

### Other ways to run

**Launcher on the host** (LocalStack from `docker compose up -d`, image from `docker build -t weyucou/agent-worker:local .`):

```bash
export SQS_QUEUE_URL=http://localhost:4566/000000000000/tasks
export SQS_ENDPOINT_URL=http://localhost:4566  # launcher -> LocalStack
export S3_ENDPOINT_URL=http://localstack:4566  # agent container -> LocalStack on the compose network
export DOCKER_NETWORK=wade_default AGENT_WORKER_IMAGE=weyucou/agent-worker:local
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1
uv run python -m marvin.launchers
```

Ctrl-C stops this launcher **and interrupts the running agent**. The terminal sends SIGINT to the whole foreground process group, which includes the `docker run` child, and `docker run` forwards it to the container. The agent exits non-zero, so its message is left and redelivered once its visibility timeout expires. To let a running task finish first, send SIGTERM to the launcher process alone (`kill <pid>`): it exits after the container does.

**Real AWS**: unset `SQS_ENDPOINT_URL` / `S3_ENDPOINT_URL`, point `SQS_QUEUE_URL` at the real queue, and either export `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`, or set `AWS_PROFILE` plus `DOCKER_AWS_CONFIG_DIR=~/.aws` (mounted read-only at `/root/.aws`; for SSO profiles run `aws sso login` first, since the container cannot save refreshed tokens). Docker has no ECS task role, so these credentials are what the agent uses.

**ECS**: `COMPUTE_BACKEND=ecs` issues `ecs.run_task` with the same `TASK_ENVELOPE_JSON` override instead of `docker run` (see [ARCHITECTURE.md](ARCHITECTURE.md#launcher-flow-marvinlaunchers)).

### Launcher settings

| Variable | Default | Purpose |
|----------|---------|---------|
| `COMPUTE_BACKEND` | `docker` | `docker` (local `docker run`) or `ecs` (`ecs:RunTask`); any other value exits with code `2` |
| `SQS_QUEUE_URL` | — | **Required.** Queue to poll |
| `SQS_ENDPOINT_URL` | — | LocalStack override for the launcher's SQS client |
| `VISIBILITY_TIMEOUT` | `300` | Seconds; the docker backend re-extends it every `VISIBILITY_TIMEOUT/2` while a container runs |
| `MAX_MESSAGES` | `1` | Messages per receive. Envelopes run one at a time; keep `1` for `docker` (higher values log a warning at startup) |
| `POLL_INTERVAL_SECONDS` | `5` | Back-off after an SQS error |
| `AGENT_WORKER_IMAGE` | `weyucou/agent-worker:latest` | Image to run (compose sets `weyucou/agent-worker:local`) |
| `DOCKER_NETWORK` | — | `docker run --network`, e.g. `wade_default` to reach `localstack:4566` |
| `DOCKER_AWS_CONFIG_DIR` | — | Host dir mounted read-only at `/root/.aws`, for `AWS_PROFILE` |
| `DOCKER_ENV_PASSTHROUGH` | — | Extra comma-separated env var names to forward to the container |
| `ECS_CLUSTER`, `ECS_TASK_DEFINITION` | — | Required for `ecs` |
| `ECS_SUBNETS`, `ECS_SECURITY_GROUPS` | — | Required for `ecs`; comma-separated (awsvpc) |
| `ECS_ASSIGN_PUBLIC_IP` | `DISABLED` | `ENABLED` or `DISABLED` |
| `ECS_CONTAINER_NAME` | `agent-worker` | Container that receives the override |

The docker backend always forwards `TASK_ENVELOPE_JSON` and `CUSTOMER_ID`, plus these when set: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_PROFILE`, `AWS_DEFAULT_REGION`, `S3_ENDPOINT_URL`, `SECRETS_MANAGER_ENDPOINT_URL`, `ANTHROPIC_API_KEY`, `GITHUB_TOKEN`, `ENVIRONMENT` (`marvin/launchers/docker.py`).

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/) package manager
- Docker Engine with the compose plugin (LocalStack, local Docker run)
- AWS CLI v2 (local Docker run)

Install commands for Ubuntu: [Setup and run on Ubuntu Linux](#setup-and-run-on-ubuntu-linux).

## Available Tools

All tools inherit from `BaseTool` (`marvin/tools/base.py`) and are registered via `register_builtin_tools()`.

### Core Tools (`marvin/tools/builtin.py`)

| Tool | Name | Description |
|------|------|-------------|
| `DateTimeTool` | `get_datetime` | Current date/time in any timezone |
| `CalculatorTool` | `calculator` | Safe arithmetic expression evaluator |
| `MemorySearchTool` | `memory_search` | Returns empty in stateless mode |

### Coding Tools (`marvin/tools/coding.py`)

| Tool | Name | `require_approval` | Description |
|------|------|-------------------|-------------|
| `ReadTool` | `read` | No | Read file contents with optional line range |
| `WriteTool` | `write` | Yes | Write/overwrite a file |
| `EditTool` | `edit` | Yes | Targeted unique-string replacement in a file |
| `ApplyPatchTool` | `apply_patch` | Yes | Apply a unified diff patch via the `patch` CLI |
| `ExecTool` | `exec` | Yes | Run shell commands (includes `gh` CLI for GitHub) |
| `ProcessTool` | `process` | Yes | Manage long-running background shell sessions |
| `WebFetchTool` | `web_fetch` | No | Fetch URL content as plain text |
| `RealWebSearchTool` | `web_search` | No | Search via Brave Search API |
| `SessionsSpawnTool` | `sessions_spawn` | Yes | Spawn a Claude CLI sub-agent session |
| `SessionsSendTool` | `sessions_send` | Yes | Send a prompt to a Claude CLI sub-agent |
| `ImageTool` | `image` | No | Analyze images via Claude vision |
| `BrowserTool` | `browser` | Yes | Browser automation via Playwright |

### Tool Profiles

| Profile | Tools |
|---------|-------|
| `minimal` | None |
| `coding` | `read`, `write`, `edit`, `apply_patch`, `exec`, `process`, `web_fetch`, `web_search`, `sessions_spawn`, `sessions_send`, `image`, `memory_search` |
| `full` | All tools |

### Required Environment Variables

| Variable | Required By | Description |
|----------|-------------|-------------|
| `SQS_QUEUE_URL` | worker | SQS queue URL to poll for tasks |
| `ZAATAR_SEARCH_API_URL` | `web_search` | Base URL of a deployed [zaatar-search-api](https://github.com/monkut/zaatar-search-api) instance (e.g. `http://localhost:5000`) |
| `ANTHROPIC_API_KEY` | `image` | Anthropic API key for vision analysis |

### Optional Runtime Dependencies

| Dependency | Required By | Install |
|------------|-------------|---------|
| `playwright` + Chromium | `browser` | `pip install playwright && playwright install chromium` |
| `claude` CLI | `sessions_spawn`, `sessions_send` | Install from [Claude Code](https://claude.ai/code) |

## Development

### Run Tests

```bash
uv run poe test
```

### Run Linter

```bash
uv run poe check
```

### Run Type Checker

```bash
uv run pyright
```

## Project Structure

```
wyc6k-task-runner/
├── marvin/                 # Stateless SQS worker
│   ├── launchers/          # SQS → one agent-worker container per envelope (docker | ecs)
│   ├── llm/                # LLM clients (Anthropic, Gemini, OpenAI, Ollama)
│   ├── tools/              # Tool base classes, registry, built-in and coding tools
│   ├── models.py           # Pydantic models (AgentConfig, TaskEnvelope)
│   ├── runner.py           # AgentRunner — prompt, tools, rate limit; runs the loop
│   ├── loop.py             # run_tool_loop — the tool-call loop for every provider
│   ├── worker.py           # SQS consumer entry point
│   ├── context.py          # ContextBundleService — reads from S3
│   └── rate_limiter.py     # Thread-safe sliding-window rate limiter
├── tests/                  # pytest tests
├── entrypoint.py           # One-shot Fargate entrypoint (agent-worker image)
├── Dockerfile              # agent-worker image (built by CI)
├── Dockerfile.launcher     # Launcher image (`local-runner` compose profile)
├── docker-compose.yaml     # LocalStack (SQS/S3); `local-runner` profile adds the launcher
└── pyproject.toml          # Project dependencies
```

## License

See [LICENSE](LICENSE) file.
