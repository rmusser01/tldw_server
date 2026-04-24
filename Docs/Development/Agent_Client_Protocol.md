# Agent Client Protocol (ACP) Module

This document summarizes the ACP integration status, layout, and how to run the
current server + runner flow. It is intended as a snapshot of progress and a
quick reference for the ACP module.

## Status Summary

- Server-side ACP client + endpoints are wired and available behind `/api/v1/acp/*`.
- **WebSocket endpoint** for real-time session streaming at `/api/v1/acp/sessions/{session_id}/stream`.
- **Permission UI flow** - Permission requests are sent to connected WebSocket clients for approval.
- ACP runner exists in `../tldw-agent` and proxies to a downstream ACP agent.
- Session lifecycle is supported: `session/new`, `session/prompt`, `session/cancel`,
  and `_tldw/session/close`.
- Downstream capabilities are reflected in `initialize`.
- Terminal tooling is allowlisted by config; file read/write is scoped to workspace.
- Tests added for the ACP runner (Go), server endpoints, and WebSocket (pytest).
- Smoke test validated via stub agent.

## Architecture

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   WebUI/Client  │────▶│  tldw_server    │────▶│   tldw-agent    │
│                 │◀────│  (FastAPI)      │◀────│   (Go Runner)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                        │
                               │                        ▼
                               │               ┌─────────────────┐
                               │               │ Downstream Agent│
                               │               │ (Claude Code/   │
                               │               │  Codex/Custom)  │
                               │               └─────────────────┘
                               │
                        REST + WebSocket
```

## Module Layout

### Server (tldw_server2)

**Core client:**
- `tldw_Server_API/app/core/Agent_Client_Protocol/stdio_client.py` - JSON-RPC stdio communication
- `tldw_Server_API/app/core/Agent_Client_Protocol/runner_client.py` - Session management, WebSocket registry, permission handling
- `tldw_Server_API/app/core/Agent_Client_Protocol/config.py` - Configuration loading

**API schemas:**
- `tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py` - Pydantic models for REST and WebSocket messages

**API endpoints:**
- `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py` - REST and WebSocket endpoints

### Runner (tldw-agent)

- `../tldw-agent/internal/acp/conn.go` - Connection management
- `../tldw-agent/internal/acp/runner.go` - ACP runner logic
- `../tldw-agent/internal/acp/terminal.go` - Terminal tool handling
- `../tldw-agent/internal/acp/stdio.go` - Stdio communication
- `../tldw-agent/internal/acp/types.go` - Type definitions
- `../tldw-agent/cmd/tldw-agent-acp/main.go` - Runner entrypoint

### Frontend (apps/packages/ui)

- `src/services/acp/types.ts` - TypeScript type definitions
- `src/services/acp/client.ts` - REST and WebSocket client
- `src/services/acp/constants.ts` - Tool tiers and configuration
- `src/hooks/useACPSession.tsx` - React hook for session management
- `src/store/acp-sessions.ts` - Zustand store for session state
- `src/components/Option/ACPPlayground/` - ACP Playground UI components

### Test Assets

- `Helper_Scripts/acp_stub_agent.py` - Stub agent for smoke testing
- `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py` - REST endpoint tests
- `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py` - WebSocket tests
- `../tldw-agent/internal/acp/runner_test.go` - Runner tests (Go)
- `../tldw-agent/internal/acp/terminal_test.go` - Terminal tests (Go)

## Endpoints

### REST Endpoints

|Endpoint|Method|Description|
|---|---|---|
|`/api/v1/acp/sessions/new`|POST|Create a new ACP session|
|`/api/v1/acp/sessions/prompt`|POST|Send a prompt to a session|
|`/api/v1/acp/sessions/cancel`|POST|Cancel the current operation|
|`/api/v1/acp/sessions/close`|POST|Close and cleanup a session|
|`/api/v1/acp/sessions/{session_id}/updates`|GET|Poll for session updates|

### WebSocket Endpoint

**URL:** `WS /api/v1/acp/sessions/{session_id}/stream`

**Query Parameters:**
- `token` - JWT access token (multi-user mode)
- `api_key` - API key (single-user mode)

**Server → Client Messages:**

|Type|Description|
|---|---|
|`connected`|Connection established, includes agent capabilities|
|`update`|Real-time update from agent session|
|`permission_request`|Tool execution requires approval|
|`error`|Error occurred|
|`prompt_complete`|Prompt execution completed|

**Client → Server Messages:**

|Type|Description|
|---|---|
|`permission_response`|Approve or deny a permission request|
|`cancel`|Cancel the current operation|
|`prompt`|Send a new prompt|

### Permission Request Example

```json
{
  "type": "permission_request",
  "request_id": "uuid",
  "session_id": "session-id",
  "tool_name": "fs.write",
  "tool_arguments": {"path": "/file.txt", "content": "..."},
  "tier": "batch",
  "timeout_seconds": 300
}
```

### Permission Response Example

```json
{
  "type": "permission_response",
  "request_id": "uuid",
  "approved": true,
  "batch_approve_tier": "batch"
}
```

## Permission Tiers

Tools are classified into permission tiers based on their risk level:

| Tier | Description | Examples |
|------|-------------|----------|
| `auto` | Auto-approved (read-only) | `fs.read`, `git.status`, `search.grep` |
| `batch` | Approve multiple at once | `fs.write`, `git.commit`, `git.add` |
| `individual` | Review each one | `fs.delete`, `exec.run`, `git.push` |

### Tier Determination Heuristics

The server automatically determines the permission tier based on the tool name using pattern matching:

**Auto tier** (read-only operations - auto-approved):
- Patterns: `read`, `get`, `list`, `search`, `find`, `view`, `show`, `glob`, `grep`, `status`
- Example: `fs.readFile` → `auto` (contains "read")

**Individual tier** (destructive operations - require individual approval):
- Patterns: `delete`, `remove`, `exec`, `run`, `shell`, `bash`, `terminal`, `push`, `force`
- Example: `git.push` → `individual` (contains "push")

**Batch tier** (default for write operations):
- Any tool that doesn't match auto or individual patterns
- Example: `fs.write` → `batch` (no special pattern match)

Pattern matching is case-insensitive and checks if the tool name contains any of the patterns.

## Governance Integration

ACP now uses a shared governance coordinator for both prompt and permission flows.

Contract details:
- Prompt checks and permission checks go through `ACPGovernanceCoordinator`.
- Permission outcome is unified to one path: `approve`, `deny`, or `prompt`.
- Governance `require_approval` is merged into the same approval prompt path as tiered ACP approvals, preventing duplicate prompts.
- Governance deny decisions raise `ACPGovernanceDeniedError` with structured governance metadata.

Compatibility and migration notes:
- MCP wire compatibility is unchanged; governance metadata is additive on MCP errors.
- ACP moves toward the unified governance contract and deprecates legacy split approval behavior.
- Rollout configuration is shared via `GOVERNANCE_ROLLOUT_MODE` (`off`, `shadow`, `enforce`).

## Configuration

### Server config.txt

Enable ACP routes in `tldw_Server_API/Config_Files/config.txt`:

```ini
[API-Routes]
stable_only = true
enable = tools, jobs, acp

[ACP]
runner_command = go
runner_args = ["run", "./cmd/tldw-agent-acp"]
runner_cwd = ../tldw-agent
runner_env = HOME=./acp_runner_home,PYTHONUNBUFFERED=1
startup_timeout_ms = 10000
```

Relative `HOME` values in `runner_env` are resolved against `tldw_Server_API/Config_Files`, so the checked-in example stays portable across machines.

### Environment Overrides

```bash
ACP_RUNNER_COMMAND=/path/to/runner
ACP_RUNNER_ARGS='["--flag","value"]'
ACP_RUNNER_ENV='HOME=/abs/path,PYTHONUNBUFFERED=1'
ACP_RUNNER_CWD=/abs/path/to/runner/dir
ACP_RUNNER_STARTUP_TIMEOUT_MS=10000
```

## ACP Sandbox Mode (Container/VM)

ACP sandbox mode runs `tldw-agent-acp` inside a sandbox container and exposes a web SSH proxy.

### Install ACP Dependencies

```bash
pip install -e ".[acp]"
```

### Build the ACP Image

With sibling repos (`../tldw-agent` next to `tldw_server2`), build from the parent directory:

```bash
# From tldw_server2/
docker build -f Dockerfiles/ACP/Dockerfile \
  --build-arg TLDW_SERVER_DIR=tldw_server2 \
  --build-arg TLDW_AGENT_DIR=tldw-agent \
  -t tldw/acp-agent:latest ..
```

### Config

Enable ACP sandbox mode and set the agent command:

```ini
[ACP-SANDBOX]
enabled = true
runtime = docker
base_image = tldw/acp-agent:latest
network_policy = allow_all
agent_command = claude
agent_args = ["code"]
```

`agent_command` must be the downstream coding agent executable (`claude`, `codex`, `opencode`, etc).  
Do not set it to `tldw-agent-acp` (that recursively launches the runner and fails with `resource temporarily unavailable`).

### Required Env

```bash
ACP_SANDBOX_ENABLED=1
ACP_SANDBOX_AGENT_COMMAND=claude
SANDBOX_ENABLE_EXECUTION=1
SANDBOX_BACKGROUND_EXECUTION=1
SANDBOX_DOCKER_BIND_WORKSPACE=1
```

### Notes

- Each ACP session starts a dedicated sandbox run.
- The container exposes SSH on a host port (local only) and the UI connects via WS proxy.

## Agent Configurations

The runner launches the downstream ACP agent based on:
`~/.tldw-agent/config.yaml` (or the HOME specified in runner_env)

### Complete Configuration Example

```yaml
# ~/.tldw-agent/config.yaml
# Complete configuration example with all available options

# Agent configuration - defines which downstream ACP agent to launch
agent:
  # Command to execute (required)
  # Can be an absolute path or command in PATH
  command: "claude"

  # Command-line arguments (optional, default: [])
  args: ["code"]

  # Environment variables for the agent process (optional)
  # Use ${VAR_NAME} to reference existing environment variables
  env:
    ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
    # Add any additional env vars needed by your agent
    # SOME_CONFIG: "value"

# Workspace configuration (optional)
workspace:
  # Allowed root directories for file operations
  # File operations outside these roots will be blocked
  allowed_roots:
    - "/home/user/projects"
    - "/tmp/sandbox"

# Terminal configuration (optional)
terminal:
  # Whether terminal tools are enabled
  enabled: true

  # Allowlist of permitted command patterns
  # Commands not matching any pattern are blocked
  allowed_commands:
    - "git *"           # Allow all git commands
    - "npm *"           # Allow npm commands
    - "python *.py"     # Allow running Python scripts
    - "ls *"            # Allow listing directories
    - "cat *"           # Allow reading files

# Logging configuration (optional)
logging:
  # Log level: debug, info, warn, error
  level: "info"

  # Log file path (optional, logs to stderr if not set)
  # file: "/var/log/tldw-agent/agent.log"
```

### Claude Code

```yaml
agent:
  command: "claude"
  args: ["code"]
  env:
    ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
```

### Codex CLI

```yaml
agent:
  command: "codex"
  args: []
  env:
    OPENAI_API_KEY: "${OPENAI_API_KEY}"
```

### Custom ACP Agent

```yaml
agent:
  command: "/path/to/custom-agent"
  args: ["--stdio"]
  env: {}
```

## Frontend Integration

### Using the useACPSession Hook

```typescript
import { useACPSession } from "@/hooks/useACPSession"

const {
  state,              // Session state
  isConnected,        // Whether connected
  updates,            // List of updates
  pendingPermissions, // Pending permission requests
  connect,            // Connect to session
  disconnect,         // Disconnect from session
  sendPrompt,         // Send a prompt
  approvePermission,  // Approve a permission
  denyPermission,     // Deny a permission
  cancel,             // Cancel current operation
} = useACPSession({
  sessionId: "session-id",
  autoConnect: true,
  onUpdate: (update) => console.log(update),
  onPermissionRequest: (request) => console.log(request),
})
```

### Using the Zustand Store

```typescript
import { useACPSessionsStore } from "@/store/acp-sessions"

const sessions = useACPSessionsStore((s) => s.getSessions())
const activeSession = useACPSessionsStore((s) =>
  s.activeSessionId ? s.getSession(s.activeSessionId) : undefined
)
const createSession = useACPSessionsStore((s) => s.createSession)
```

## Testing

### Server Endpoint Tests

```bash
python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py -v
```

### WebSocket Tests

```bash
python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py -v
```

### Runner Tests (Go)

```bash
cd ../tldw-agent
go test ./internal/acp -v
```

## Behavior Summary

### Server ACP Client

- Spawns the ACP runner via stdio and JSON-RPC line framing.
- Maintains a per-process client and queues session updates.
- **WebSocket Registry**: Tracks connected WebSocket clients per session.
- **Permission Flow**: Permission requests are broadcast to connected WebSocket clients.
  - If no WebSocket is connected, permissions are auto-cancelled after 5 minutes.
  - Auto-approve for `auto` tier tools.
  - Batch approval option for `batch` tier tools.
- Supports env/config overrides for runner command/args/env/cwd.

### ACP Runner

- One downstream process per ACP session.
- Validates workspace roots and keeps file ops inside workspace.
- Handles allowlisted `terminal/*` tools with command templates and argument guards.
- Forwards `session/request_permission` upstream.
- Caches downstream capabilities and reflects them in `initialize`.

## Known Constraints

- ACP routes are gated by `stable_only` unless explicitly enabled.
- ACP runner uses stdio; ensure the runner executable is available in PATH or
  configured explicitly.
- Permission timeout is 5 minutes; requests are auto-cancelled if not responded to.
- WebSocket reconnection uses exponential backoff (max 10 attempts).

## Troubleshooting

### Connection Issues

1. **Check ACP is enabled**: Verify `enable = acp` in config.txt `[API-Routes]` section.
2. **Check runner path**: Ensure `runner_command` points to a valid executable.
3. **Check authentication**: Verify JWT token or API key is valid.

### Permission Requests Not Appearing

1. **WebSocket connected**: Ensure the WebSocket connection is established before sending prompts.
2. **Tool tier**: `auto` tier tools are auto-approved and won't trigger permission requests.

### Agent Not Responding

1. **Check agent config**: Verify `~/.tldw-agent/config.yaml` has correct agent configuration.
2. **Check API keys**: Ensure ANTHROPIC_API_KEY or OPENAI_API_KEY is set for the agent.
3. **Check logs**: Review server logs for ACP-related errors.
