# Getting Started with ACP (Agent Client Protocol)

This guide helps you connect your local coding agents (Claude Code, Codex, OpenCode) to tldw_server so you can control them remotely from the web UI.

For Anthropic-specific first-time setup patterns (BYOK + Claude Code/SDK), see `Docs/User_Guides/Integrations_Experiments/Anthropic_ClaudeCode_ClaudeSDK_Setup.md`.

## What is ACP?

ACP (Agent Client Protocol) lets you control AI coding assistants from your browser. Instead of running Claude Code directly in your terminal, you can:

- Send prompts from the web UI
- Approve or deny file changes visually
- Monitor agent activity in real-time
- Switch between multiple sessions

**Architecture Overview:**

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Browser       │────▶│  tldw_server    │────▶│   tldw-agent    │
│   (WebUI)       │◀────│  (FastAPI)      │◀────│   (Go Runner)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │ Downstream Agent│
                                                │ (Claude Code/   │
                                                │  Codex/Custom)  │
                                                └─────────────────┘
```

The flow:
1. You type a prompt in the browser
2. tldw_server sends it to tldw-agent (Go runner)
3. tldw-agent launches your coding agent (e.g., Claude Code)
4. Results stream back through tldw_server to your browser
5. Permission requests appear in the UI for you to approve/deny

## Prerequisites

Before starting, ensure you have:

- **tldw_server running** - See [Self-Hosting Profiles](../../Getting_Started/README.md)
- **Go 1.22+** - Required for building tldw-agent ([go.dev/dl](https://go.dev/dl))
- **Claude Code installed** - Or another ACP-compatible agent
- **Anthropic API key** - For Claude Code (get one at [console.anthropic.com](https://console.anthropic.com))

## Step-by-Step Setup

### Step 1: Enable ACP in tldw_server

Edit `tldw_Server_API/Config_Files/config.txt` and ensure ACP is enabled.

**Minimal config (required settings only):**

```ini
[API-Routes]
stable_only = false

[ACP]
runner_command = /path/to/tldw-agent-acp
runner_cwd = /path/to/tldw-agent
```

**Full config (with all options):**

```ini
[API-Routes]
stable_only = false
enable = tools, jobs, acp

[ACP]
runner_command = go
runner_args = ["run", "./cmd/tldw-agent-acp"]
runner_cwd = ../tldw-agent
runner_env = HOME=./acp_runner_home,PYTHONUNBUFFERED=1
startup_timeout_ms = 10000
```

Relative `HOME` values in `runner_env` are resolved against `tldw_Server_API/Config_Files`.

Install ACP dependencies:

```bash
pip install -e ".[acp]"
```

### Optional: Sandbox Mode (Run ACP in Containers)

To run the ACP agent inside a sandbox container and access it via web SSH:

1. Build the ACP image:

```bash
# From tldw_server2/ with sibling ../tldw-agent
docker build -f Dockerfiles/ACP/Dockerfile \
  --build-arg TLDW_SERVER_DIR=tldw_server2 \
  --build-arg TLDW_AGENT_DIR=tldw-agent \
  -t tldw/acp-agent:latest ..
```

2. Enable sandbox mode in `config.txt`:

```ini
[ACP-SANDBOX]
enabled = true
runtime = docker
base_image = tldw/acp-agent:latest
network_policy = allow_all
agent_command = claude
agent_args = ["code"]
```

`agent_command` must point to the downstream coding agent binary (`claude`, `codex`, `opencode`, etc.).  
Do not set `agent_command` to `tldw-agent-acp` because that recursively launches the runner and will fail with `resource temporarily unavailable`.

3. Set required env vars:

```bash
export ACP_SANDBOX_ENABLED=1
export ACP_SANDBOX_AGENT_COMMAND=claude
export SANDBOX_ENABLE_EXECUTION=1
export SANDBOX_BACKGROUND_EXECUTION=1
export SANDBOX_DOCKER_BIND_WORKSPACE=1
```

Start `tldw_server` from the same shell (or ensure your service manager passes these env vars) so the server process inherits the sandbox settings.

**Key settings explained:**

| Setting | Description |
|---------|-------------|
| `runner_command` | The executable to run (`go` for development, or path to compiled binary) |
| `runner_args` | Arguments passed to the command |
| `runner_cwd` | Working directory where tldw-agent is located |
| `runner_env` | Environment variables for the runner (HOME points to agent config location) |
| `startup_timeout_ms` | How long to wait for the runner to initialize |

**Development vs Production:**

For development (using `go run`):
```ini
runner_command = go
runner_args = ["run", "./cmd/tldw-agent-acp"]
runner_cwd = ../tldw-agent
```

For production (pre-built binary):
```ini
runner_command = /opt/tldw-agent/bin/tldw-agent-acp
runner_args = []
runner_cwd = /opt/tldw-agent
```

### Docker Networking for ACP

If you run tldw_server or tldw-agent inside Docker, the two processes need to reach each other over the network. Below are the two most common setups.

**Scenario 1: Server in Docker, Runner on Host**

The runner (tldw-agent) runs on the host and listens on a local port. Add `extra_hosts` so the container can reach the host network:

```yaml
# docker-compose.yml
services:
  tldw-server:
    image: tldw/server:latest
    ports:
      - "8000:8000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      ACP_RUNNER_COMMAND: "http://host.docker.internal:9090"
```

Then start tldw-agent on the host normally. The server container will reach it via `host.docker.internal`.

**Scenario 2: Both in Docker**

Run tldw-agent as a sibling service in the same Compose project so they share a Docker network:

```yaml
# docker-compose.yml
services:
  tldw-server:
    image: tldw/server:latest
    ports:
      - "8000:8000"
    environment:
      ACP_RUNNER_COMMAND: "http://tldw-agent:9090"
    depends_on:
      - tldw-agent

  tldw-agent:
    build:
      context: ../tldw-agent
      dockerfile: Dockerfile
    environment:
      ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
    expose:
      - "9090"
```

Both services join the default Compose network, so the server can reach the runner at `http://tldw-agent:9090`.

### Step 2: Set Up tldw-agent (the Runner)

Clone and build the tldw-agent repository:

```bash
# Clone the repository (sibling to tldw_server2)
cd ..
git clone https://github.com/rmusser01/tldw-agent.git
cd tldw-agent
# Note: This repository may not yet be public. If the clone fails,
# contact the maintainer or check for build-from-source instructions below.

# Build the binary
go build -o bin/tldw-agent-acp ./cmd/tldw-agent-acp

# Verify it built correctly
./bin/tldw-agent-acp --help
```

Create the config directory:

```bash
mkdir -p ~/.tldw-agent
```

### Step 3: Configure for Claude Code

Create `~/.tldw-agent/config.yaml` with your Claude Code settings:

```yaml
# ~/.tldw-agent/config.yaml
agent:
  command: "claude"
  args: ["code"]
  env:
    ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"

# Optional: Restrict file operations to specific directories
workspace:
  allowed_roots:
    - "/home/user/projects"
    - "/tmp/sandbox"

# Optional: Control which terminal commands are allowed
terminal:
  enabled: true
  allowed_commands:
    - "git *"
    - "npm *"
    - "python *.py"
    - "ls *"
    - "cat *"

logging:
  level: "info"
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Or add it to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.).

### Step 4: Test the Connection

1. **Start tldw_server:**
   ```bash
   cd /path/to/tldw_server2
   python -m uvicorn tldw_Server_API.app.main:app --reload
   ```

2. **Verify ACP endpoints are available:**
   ```bash
   # Should return available agents
   curl -s http://127.0.0.1:8000/api/v1/acp/agents \
     -H "X-API-KEY: your-api-key" | jq
   ```

3. **Open the WebUI:**
   Navigate to the ACP Playground page in your browser.

## Your First Session

### Creating a Session

1. Open the ACP Playground in the WebUI
2. Click "New Session"
3. Select your agent type (Claude Code)
4. Optionally set a working directory (cwd)
5. Click "Create"

Or via API:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/acp/sessions/new \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "claude_code",
    "cwd": "/path/to/your/project",
    "name": "My First Session"
  }'
```

### Sending a Prompt

Type your prompt in the input field and press Enter. For example:

> "List all Python files in this directory"

The agent will process your request and stream results back in real-time.

### Understanding Permission Requests

When the agent wants to perform certain actions, you'll see permission requests:

**Permission Tiers:**

| Tier | Description | Examples | Approval |
|------|-------------|----------|----------|
| `auto` | Read-only operations | Reading files, git status, searching | Auto-approved |
| `batch` | Write operations | Writing files, git commit | Can approve multiple at once |
| `individual` | Destructive operations | Deleting files, git push, running scripts | Must approve each one |

When a permission request appears:
- **Approve**: Click "Approve" to allow the action
- **Approve All (Batch)**: For batch-tier requests, approve all similar pending requests
- **Deny**: Click "Deny" to reject the action

**Timeout:** Permission requests expire after 5 minutes if not responded to.

### Closing a Session

When you're done:
1. Click "Close Session" in the UI
2. Or via API:
   ```bash
   curl -X POST http://127.0.0.1:8000/api/v1/acp/sessions/close \
     -H "X-API-KEY: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "your-session-id"}'
   ```

## Alternative Agents

### Codex CLI (OpenAI)

```yaml
# ~/.tldw-agent/config.yaml
agent:
  command: "codex"
  args: []
  env:
    OPENAI_API_KEY: "${OPENAI_API_KEY}"
```

### OpenCode

```yaml
# ~/.tldw-agent/config.yaml
agent:
  command: "opencode"
  args: []
  env: {}
```

### Custom ACP Agent

```yaml
# ~/.tldw-agent/config.yaml
agent:
  command: "/path/to/your/agent"
  args: ["--stdio", "--mode", "acp"]
  env:
    CUSTOM_VAR: "value"
```

## Troubleshooting

### Quick Troubleshooting Checklist

Work through these steps in order. Stop at the first failure and apply the fix.

**1. Can you reach the server?**

- Test: `curl http://127.0.0.1:8000/docs`
- If no: Start tldw_server (`python -m uvicorn tldw_Server_API.app.main:app --reload`) and check for startup errors in the console.

**2. Are ACP routes enabled?**

- Test: `curl -s http://127.0.0.1:8000/api/v1/acp/health -H "X-API-KEY: <YOUR_API_KEY>"`
- If no (404): Set `stable_only = false` in `[API-Routes]` in `config.txt` and restart the server.

**3. Is the runner configured?**

- Test: Check the health response from step 2 — it should show runner status.
- If no: Verify `[ACP] runner_command` and `runner_cwd` are set correctly in `config.txt`. See the config examples above.

**4. Is the downstream agent installed?**

- Test: `claude --version` (or `codex --version`, `opencode --version`)
- If no: Install your chosen agent. For Claude Code see [claude.ai/download](https://claude.ai/download).

**5. Is the API key set?**

- Test: `echo $ANTHROPIC_API_KEY`
- If no: `export ANTHROPIC_API_KEY=sk-ant-...` and add it to `~/.tldw-agent/config.yaml` under `agent.env`.

**6. Can you create a session?**

- Test:
  ```bash
  curl -X POST http://127.0.0.1:8000/api/v1/acp/sessions/new \
    -H "X-API-KEY: <YOUR_API_KEY>" \
    -H "Content-Type: application/json" \
    -d '{"agent_type": "claude_code", "cwd": "/tmp"}'
  ```
- If no: Check server logs for the specific error. Common causes include incorrect `runner_command` path or missing Go installation.

---

### "ACP endpoints not found" (404)

**Cause:** ACP routes are not enabled.

**Fix:** Edit `config.txt` and ensure:
```ini
[API-Routes]
stable_only = false
enable = tools, jobs, acp
```

Then restart tldw_server.

### "Runner failed to start"

**Cause:** The runner command or path is incorrect.

**Fix:**
1. Verify Go is installed: `go version`
2. Check the runner path exists: `ls ../tldw-agent/cmd/tldw-agent-acp`
3. Try running manually:
   ```bash
   cd ../tldw-agent
   go run ./cmd/tldw-agent-acp
   ```
4. Check server logs for specific error messages

### "Agent not responding"

**Cause:** Agent configuration issue or missing API key.

**Fix:**
1. Verify your config file exists: `cat ~/.tldw-agent/config.yaml`
2. Check the agent command works directly:
   ```bash
   claude code --help
   ```
3. Ensure API key is set:
   ```bash
   echo $ANTHROPIC_API_KEY
   ```
4. Check the HOME environment in `runner_env` points to the config directory. Relative values resolve from `tldw_Server_API/Config_Files`.

### WebSocket Connection Fails

**Cause:** Authentication issue or network problem.

**Fix:**
1. Ensure you're authenticated (check your API key or JWT token)
2. Check browser console for specific errors
3. Verify the WebSocket URL is correct: `ws://127.0.0.1:8000/api/v1/acp/sessions/{session_id}/stream`
4. If using HTTPS, ensure WebSocket uses `wss://`

### Permission Requests Not Appearing

**Cause:** WebSocket not connected, or tool is auto-approved.

**Fix:**
1. Verify WebSocket connection status in the UI
2. Remember that `auto` tier tools (read operations) don't show permission requests
3. Check server logs for permission-related messages

### "Workspace root not allowed"

**Cause:** The agent tried to access files outside allowed directories.

**Fix:** Update `~/.tldw-agent/config.yaml`:
```yaml
workspace:
  allowed_roots:
    - "/path/to/your/project"
    - "/another/allowed/path"
```

## Security Notes

### Workspace Restrictions

Configure `allowed_roots` in your agent config to limit which directories the agent can access. This prevents accidental modifications to system files.

### Command Allowlisting

Use `terminal.allowed_commands` to restrict which shell commands the agent can execute:

```yaml
terminal:
  enabled: true
  allowed_commands:
    - "git *"          # Allow git commands
    - "npm install"    # Specific npm command only
    - "python test.py" # Specific script only
```

Commands not matching any pattern are blocked.

### Permission Timeout

Permission requests automatically expire after 5 minutes. This prevents stale requests from being accidentally approved later.

### Environment Variables

Sensitive values like API keys can use environment variable interpolation:
```yaml
env:
  ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
```

This keeps secrets out of config files.

## Next Steps

- **Technical Reference**: See [Agent Client Protocol](../../Development/Agent_Client_Protocol.md) for detailed API documentation
- **WebSocket Integration**: Learn about real-time streaming and the WebSocket message protocol
- **Frontend Development**: Explore the React hooks and Zustand store for building custom UIs

## Quick Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/acp/sessions/new` | POST | Create a new session |
| `/api/v1/acp/sessions/prompt` | POST | Send a prompt |
| `/api/v1/acp/sessions/cancel` | POST | Cancel current operation |
| `/api/v1/acp/sessions/close` | POST | Close a session |
| `/api/v1/acp/sessions/{id}/updates` | GET | Poll for updates |

### WebSocket

**URL:** `WS /api/v1/acp/sessions/{session_id}/stream`

**Authentication:** Pass `token` (JWT) or `api_key` as query parameter.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ACP_RUNNER_COMMAND` | Override runner command |
| `ACP_RUNNER_ARGS` | Override runner arguments (JSON array) |
| `ACP_RUNNER_CWD` | Override runner working directory |
| `ACP_RUNNER_ENV` | Override runner environment |
| `ACP_RUNNER_STARTUP_TIMEOUT_MS` | Override startup timeout |
