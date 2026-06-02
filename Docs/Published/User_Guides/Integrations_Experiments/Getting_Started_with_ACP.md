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

Edit `tldw_Server_API/Config_Files/config.txt` and ensure ACP is enabled:

```ini
[API-Routes]
stable_only = false
enable = tools, jobs, acp

[ACP]
runner_command = go
runner_args = ["run", "./cmd/tldw-agent-acp"]
runner_cwd = ../tldw-agent
runner_env = HOME=/absolute/path/to/.tldw-agent-home,PYTHONUNBUFFERED=1
startup_timeout_ms = 10000
```

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

### Step 2: Set Up tldw-agent (the Runner)

Clone and build the tldw-agent repository:

```bash
# Clone the repository (sibling to tldw_server2)
cd ..
git clone https://github.com/your-org/tldw-agent.git
cd tldw-agent

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

Codex uses an external ACP adapter profile. Install the OpenAI Codex CLI
separately first, then install `zed-industries/codex-acp` `0.15.0` from the
GitHub release artifact and ensure both `codex` and `codex-acp` are on PATH.

For operator setup or certification environments where release artifacts are not
available, a pinned npm install of `@zed-industries/codex-acp@0.15.0` is an
acceptable alternative. Do not use `@latest` for certification or seeded runtime
configuration.

Passive readiness checks only inspect configured binaries and metadata. They do
not install packages, invoke package managers, or run `npx @latest`.

Codex CLI `0.128.0` through `codex-acp` `0.15.0` has backend live E2E
coverage on the macOS host runner for health/setup-guide, session creation,
prompting, redacted support views, diagnostics, cancel, and close. Sandbox,
non-empty MCP injection, artifact-producing workflows, and reviewer-loop
behavior remain unverified.

```yaml
# tldw_Server_API/Config_Files/agents.yaml
- type: codex
  command: codex
  entrypoint_strategy: external_acp_adapter
  acp_command: codex-acp
  acp_args: []
  adapter_source: zed-industries/codex-acp
  adapter_version: "0.15.0"
  adapter_version_policy: exact_pin_required
  adapter_install_source: github_release_preferred
  credential_policy: delegated_to_adapter
  support_state: supported_with_caveats
  verification_level: live_e2e_tested
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
   For Codex adapter setups, also check `codex --version` and `codex-acp --version`.
3. Ensure API key is set:
   ```bash
   echo $ANTHROPIC_API_KEY
   ```
4. Check the HOME environment in `runner_env` points to the config directory

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
