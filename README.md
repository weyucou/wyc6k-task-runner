# wyc6k-task-runner

The **agent runtime** of the WYC6k multi-tenant agent harness. Dequeues `TaskEnvelope` messages from SQS, runs an LLM agent in a tool-call loop against a GitHub issue, and writes memory back to S3 on completion.

## System Architecture

wyc6k-task-runner is the agent runtime of the **WYC6k** multi-tenant agent harness. See [weyucou/wyc6k-spec](https://github.com/weyucou/wyc6k-spec) for the full system architecture, and [ARCHITECTURE.md](ARCHITECTURE.md) for wyc6k-task-runner internals.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/monkut/marvin-manager.git
cd marvin-manager
uv sync

# 2. Start PostgreSQL (using Docker)
docker-compose up -d

# 3. Run the setup wizard
cd mrvn
uv run python manage.py onboard

# 4. Start the development server
uv run python manage.py runserver
```

Access the admin panel at http://127.0.0.1:8000/admin/

## Features

- **Multi-Channel Messaging**: Telegram and Slack integration
- **Multiple LLM Providers**: Anthropic, Google Gemini, Ollama, vLLM
- **Conversation Memory**: Session persistence and message history
- **Auto-Reply**: Configurable routing rules and response logic
- **Rate Limiting**: Built-in rate limiting to avoid API throttling
- **Django Admin**: Full management interface
- **Rich Tool Library**: File I/O, shell execution, web access, image analysis, browser automation, and sub-agent sessions

## Available Tools

All tools inherit from `BaseTool` (`mrvn/agents/tools/base.py`) and are registered via `register_builtin_tools()`.

### Core Tools (`mrvn/agents/tools/builtin.py`)

| Tool | Name | Description |
|------|------|-------------|
| `DateTimeTool` | `get_datetime` | Current date/time in any timezone |
| `CalculatorTool` | `calculator` | Safe arithmetic expression evaluator |
| `MemoryStoreTool` | `memory_store` | Store key-value pairs in session memory |
| `MemoryRetrieveTool` | `memory_retrieve` | Retrieve stored values by key |
| `MemorySearchTool` | `memory_search` | Hybrid vector+text search over conversation history |

### Coding Tools (`mrvn/agents/tools/coding.py`)

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

Agents are assigned a `ToolProfile` controlling which tools they can access:

| Profile | Tools |
|---------|-------|
| `minimal` | None |
| `coding` | `read`, `write`, `edit`, `apply_patch`, `exec`, `process`, `web_fetch`, `web_search`, `sessions_spawn`, `sessions_send`, `image`, `memory_store`, `memory_retrieve`, `memory_search` |
| `messaging` | Tools with messaging prefixes |
| `full` | All tools |

### Required Environment Variables for Coding Tools

| Variable | Required By | Description |
|----------|-------------|-------------|
| `BRAVE_SEARCH_API_KEY` | `web_search` | Brave Search API key (https://api.search.brave.com/) |
| `ANTHROPIC_API_KEY` | `image` | Anthropic API key for vision analysis |

### Optional Runtime Dependencies for Coding Tools

| Dependency | Required By | Install |
|------------|-------------|---------|
| `playwright` + Chromium | `browser` | `pip install playwright && playwright install chromium` |
| `claude` CLI | `sessions_spawn`, `sessions_send` | Install from [Claude Code](https://claude.ai/code) |

## Requirements

- Python 3.14
- PostgreSQL 14+
- [uv](https://docs.astral.sh/uv/) package manager

## Ubuntu/Linux Setup

### 1. Install System Dependencies

```bash
# Update package list
sudo apt update

# Install build essentials and PostgreSQL
sudo apt install -y build-essential libpq-dev postgresql postgresql-contrib

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell or source the path
source ~/.bashrc  # or ~/.zshrc
```

### 2. Install Python 3.14

```bash
# uv can install Python for you
uv python install 3.14
```

### 3. Clone and Setup Project

```bash
# Clone the repository
git clone https://github.com/monkut/marvin-manager.git
cd marvin-manager

# Install dependencies
uv sync

# Setup pre-commit hooks (optional, for development)
uv tool install pre-commit
pre-commit install
```

### 4. Configure PostgreSQL

```bash
# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql <<EOF
CREATE USER mrvn WITH PASSWORD 'your_secure_password';
CREATE DATABASE mrvn OWNER mrvn;
GRANT ALL PRIVILEGES ON DATABASE mrvn TO mrvn;
EOF
```

### 5. Environment Configuration

Create a `.env` file or export environment variables:

```bash
# Database
export DB_NAME=mrvn
export DB_USER=mrvn
export DB_PASS=your_secure_password
export DB_HOST=127.0.0.1
export DB_PORT=5432

# Django
export DEBUG=True
export IS_LOCAL_DEVELOPMENT=True
export SECRET_KEY='your-secret-key-here'

# Rate Limiting (optional)
export RATE_LIMIT_ENABLED=True
export RATE_LIMIT_DEFAULT_RPM=60
```

### 6. Initialize the Application

**Option A: Interactive Setup Wizard (Recommended)**

```bash
cd mrvn
uv run python manage.py onboard
```

The wizard will:
- Check system requirements
- Run database migrations
- Create an admin superuser
- Configure a default LLM agent

**Option B: Manual Setup**

```bash
cd mrvn
uv run python manage.py migrate
uv run python manage.py createsuperuser
```

### 7. Run Development Server

```bash
uv run python manage.py runserver
```

Access the admin at: http://127.0.0.1:8000/admin/

## LLM Provider Configuration

### Anthropic (Claude)

1. Get API key from https://console.anthropic.com/
2. In Django Admin, create an Agent with:
   - Provider: `anthropic`
   - Model: `claude-sonnet-4-20250514` (or other Claude model)
   - Add credentials with your API key

### Google Gemini

1. Get API key from https://aistudio.google.com/
2. In Django Admin, create an Agent with:
   - Provider: `gemini`
   - Model: `gemini-2.0-flash`
   - Add credentials with your API key

### Ollama (Local)

1. Install Ollama: https://ollama.ai/download
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull llama3.2
   ```
2. In Django Admin, create an Agent with:
   - Provider: `ollama`
   - Model: `llama3.2`
   - Base URL: `http://localhost:11434`
   - No credentials needed

### vLLM (Local)

1. Install and run vLLM:
   ```bash
   pip install vllm
   vllm serve meta-llama/Llama-3.2-8B --port 8000
   ```
2. In Django Admin, create an Agent with:
   - Provider: `vllm`
   - Model: `meta-llama/Llama-3.2-8B`
   - Base URL: `http://localhost:8000/v1`

## Channel Configuration

### Telegram

**Option A: Using the Setup Command (Recommended)**

```bash
cd mrvn
uv run python manage.py setup_telegram --owner <your-username>
```

The command will guide you through:
- Entering your bot token from [@BotFather](https://t.me/BotFather)
- Configuring webhook or polling mode
- Validating the bot connection

**Option B: Manual Setup via Django Admin**

1. Create a bot via [@BotFather](https://t.me/BotFather)
2. In Django Admin, create a Channel with type `telegram`
3. Add credentials with your bot token

**Running the Telegram Bot**

```bash
# Development (foreground)
uv run python manage.py run_telegram

# Production (via systemd)
sudo systemctl start marvin-telegram
```

### Slack

**Option A: Using the Setup Command (Recommended)**

```bash
cd mrvn
uv run python manage.py setup_slack --owner <your-username>
```

The command will guide you through:
- Entering your bot token (`xoxb-...`)
- Entering your signing secret
- Optionally configuring Socket Mode with app token (`xapp-...`)

**Option B: Manual Setup via Django Admin**

1. Create a Slack App at https://api.slack.com/apps
2. Enable Socket Mode and get tokens
3. In Django Admin, create a Channel with type `slack`
4. Add credentials with bot token and signing secret

**Running the Slack Bot**

```bash
# Development (foreground, requires Socket Mode)
uv run python manage.py run_slack

# Production (via systemd)
sudo systemctl start marvin-slack
```

### Required Slack Permissions

When configuring your Slack App, add these bot scopes:
- `chat:write` - Send messages
- `app_mentions:read` - Respond to @mentions
- `im:history` - Read direct messages
- `im:read` - Access direct message channels

## Management Commands

| Command | Description |
|---------|-------------|
| `onboard` | Interactive setup wizard for initial configuration |
| `setup_telegram` | Configure a Telegram bot channel |
| `setup_slack` | Configure a Slack bot channel |
| `run_telegram` | Run the Telegram bot daemon |
| `run_slack` | Run the Slack bot daemon |

**Examples:**

```bash
cd mrvn

# Initial setup
uv run python manage.py onboard

# Setup channels
uv run python manage.py setup_telegram --owner admin
uv run python manage.py setup_slack --owner admin

# Run bots (development)
uv run python manage.py run_telegram
uv run python manage.py run_slack

# Non-interactive mode (for automation/CI)
uv run python manage.py onboard --non-interactive
uv run python manage.py setup_telegram --owner admin --bot-token "123:ABC" --non-interactive
```

## Development

### Add New Packages

```bash
uv add <package-name>
```

### Run Linters

```bash
uv run poe check
```

### Run Type Checker

```bash
uv run pyright
```

### Run Tests

```bash
uv run poe test
```

## Docker Setup (Alternative)

```bash
# Start PostgreSQL with Docker Compose
docker-compose up -d

# Run migrations
cd mrvn && uv run python manage.py migrate
```

## Production Deployment

### System Requirements

- Ubuntu 20.04+ or Debian 11+
- Python 3.14+
- PostgreSQL 14+
- systemd (for service management)

### Quick Install

```bash
# Clone repository and run install script
git clone https://github.com/monkut/marvin-manager.git
cd marvin-manager
sudo bash deploy/install.sh
```

### Manual Installation

#### 1. Create System User

```bash
sudo useradd --system --home-dir /opt/marvin --shell /bin/bash marvin
sudo mkdir -p /opt/marvin /var/log/marvin
sudo chown -R marvin:marvin /opt/marvin /var/log/marvin
```

#### 2. Install Application

```bash
# Clone to /opt/marvin
sudo -u marvin git clone https://github.com/monkut/marvin-manager.git /opt/marvin

# Install dependencies
cd /opt/marvin
sudo -u marvin uv sync
```

#### 3. Configure Environment

```bash
# Copy environment template
sudo cp /opt/marvin/deploy/marvin.env.example /opt/marvin/.env
sudo chmod 600 /opt/marvin/.env
sudo chown marvin:marvin /opt/marvin/.env

# Edit with your settings
sudo nano /opt/marvin/.env
```

Key settings to configure:
- `SECRET_KEY` - Generate with: `python -c "import secrets; print(secrets.token_urlsafe(50))"`
- `DB_PASS` - Your PostgreSQL password
- `ALLOWED_HOSTS` - Your domain name(s)

#### 4. Initialize Database

```bash
cd /opt/marvin/mrvn
sudo -u marvin uv run python manage.py onboard --non-interactive
```

#### 5. Install Systemd Services

```bash
# Copy service files
sudo cp /opt/marvin/deploy/marvin-web.service /etc/systemd/system/
sudo cp /opt/marvin/deploy/marvin-telegram.service /etc/systemd/system/
sudo cp /opt/marvin/deploy/marvin-slack.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload
```

#### 6. Start Services

```bash
# Enable and start web server
sudo systemctl enable --now marvin-web

# (Optional) Enable Telegram bot
sudo systemctl enable --now marvin-telegram

# (Optional) Enable Slack bot
sudo systemctl enable --now marvin-slack
```

### Nginx Reverse Proxy (Recommended)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /opt/marvin/mrvn/static/;
    }
}
```

### Service Management

```bash
# View status
sudo systemctl status marvin-web
sudo systemctl status marvin-telegram
sudo systemctl status marvin-slack

# View logs
sudo journalctl -u marvin-web -f
sudo journalctl -u marvin-telegram -f
sudo journalctl -u marvin-slack -f

# Restart services
sudo systemctl restart marvin-web
sudo systemctl restart marvin-telegram
sudo systemctl restart marvin-slack
```

### Updating

```bash
cd /opt/marvin
sudo -u marvin git pull
sudo -u marvin uv sync
cd mrvn && sudo -u marvin uv run python manage.py migrate
sudo systemctl restart marvin-web marvin-telegram marvin-slack
```

## Project Structure

```
marvin-manager/
├── mrvn/                   # Django application
│   ├── mrvn/               # Django project settings
│   ├── accounts/           # User authentication
│   ├── agents/             # LLM agent configuration
│   ├── channels/           # Messaging integrations (Telegram, Slack)
│   ├── memory/             # Conversation storage and sessions
│   ├── autoreply/          # Response routing and rules
│   └── commons/            # Shared utilities and rate limiting
├── deploy/                 # Production deployment files
│   ├── marvin-web.service  # Systemd service for web server
│   ├── marvin-telegram.service
│   ├── marvin-slack.service
│   ├── marvin.env.example  # Environment template
│   └── install.sh          # Installation script
├── docker-compose.yaml     # PostgreSQL for development
└── pyproject.toml          # Project dependencies
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_NAME` | `mrvn` | PostgreSQL database name |
| `DB_USER` | `postgres` | PostgreSQL username |
| `DB_PASS` | `mysecretpassword` | PostgreSQL password |
| `DB_HOST` | `127.0.0.1` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DEBUG` | `False` | Django debug mode |
| `SECRET_KEY` | (generated) | Django secret key |
| `RATE_LIMIT_ENABLED` | `True` | Enable rate limiting |
| `RATE_LIMIT_DEFAULT_RPM` | `60` | Default requests per minute |

## License

See [LICENSE](LICENSE) file.
