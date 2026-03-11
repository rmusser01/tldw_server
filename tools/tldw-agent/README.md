# tldw-agent

Local workspace MCP server for the tldw browser extension. Provides agentic coding/writing assistance similar to Claude Code or Codex.

## Overview

tldw-agent is a native messaging host that bridges your browser extension to local workspace tools. It runs entirely on your machine - your code never leaves your computer unless you explicitly share it.

### Features

- **Filesystem operations**: List, read, write files and apply patches
- **Code search**: Fast grep and glob with ripgrep-like behavior
- **Git integration**: Status, diff, log, branch, add, commit
- **Command execution**: Allowlisted commands for tests, linters, formatters
- **Cross-platform**: macOS, Linux, Windows support
- **Cross-browser**: Chrome, Firefox, Edge

## Architecture

```
Browser Extension  ←→  Native Messaging  ←→  tldw-agent  ←→  Local Filesystem
                                                         ←→  Git
                                                         ←→  CLI Tools

                   LLM calls only
                        ↓
               tldw_server (local/remote)
```

## Installation

### Prerequisites

- Go 1.22 or later
- Git

### Build

```bash
# Clone the repository
git clone https://github.com/tldw/tldw-agent.git
cd tldw-agent

# Build the binary
go build -o bin/tldw-agent-host ./cmd/tldw-agent-host
```

## Local Verification In `tldw_server2`

This tree is now also used for the `vz_linux` helper/guest roadmap inside
`tldw_server2`.

During that work, existing host/native-messaging and ACP behavior must keep
building and testing cleanly before any guest-mode changes are accepted.

Use the repo-local verification script:

```bash
bash ./scripts/verify-local-build.sh
```

The script builds into a temporary directory and runs `go test ./...`, so it
does not recreate compiled binaries inside the source tree.

### Install Native Messaging Host

**macOS/Linux (Chrome):**
```bash
./scripts/install-chrome.sh
```

**macOS/Linux (Firefox):**
```bash
./scripts/install-firefox.sh
```

**Windows:**
```powershell
.\scripts\install-windows.ps1
```

## Configuration

Configuration is stored at `~/.tldw-agent/config.yaml`:

```yaml
server:
  llm_endpoint: "http://localhost:8000"
  api_key: ""

workspace:
  default_root: ""
  blocked_paths:
    - ".env"
    - "*.pem"
    - "*.key"
    - "**/node_modules/**"
  max_file_size_bytes: 10000000

execution:
  enabled: true
  timeout_ms: 30000
  shell: "auto"
  network_allowed: false

security:
  require_approval_for_writes: true
  require_approval_for_exec: true
  redact_secrets: true

# Optional ACP agent registry (preferred for ACP sessions)
agents:
  default: "pi-agent"
  agents:
    - type: "pi-agent"
      name: "PI Agent"
      description: "First-party ACP agent based on pi-mono"
      command: "node"
      args:
        - "/absolute/path/to/pi-mono/packages/agent-acp/dist/cli.js"
      env:
        - "OPENAI_BASE_URL=http://127.0.0.1:8000/api/v1"
        - "OPENAI_API_KEY=your-api-key"

# Legacy single-agent config (used if agents registry is empty)
agent:
  command: ""
  args: []
  env: []
```

## Available Tools

### Tier 0: Read-only (auto-approve)

| Tool | Description |
|------|-------------|
| `workspace.list` | List registered workspaces |
| `workspace.pwd` | Get current working directory |
| `workspace.chdir` | Change working directory |
| `fs.list` | List directory contents |
| `fs.read` | Read file contents |
| `search.grep` | Search file contents (regex) |
| `search.glob` | Find files by pattern |
| `git.status` | Repository status |
| `git.diff` | Show changes |
| `git.log` | Recent commits |
| `git.branch` | Branch information |

### Tier 1: Write (requires approval)

| Tool | Description |
|------|-------------|
| `fs.write` | Write content to file |
| `fs.apply_patch` | Apply unified diff |
| `fs.mkdir` | Create directory |
| `fs.delete` | Delete file/directory |
| `git.add` | Stage files |
| `git.commit` | Create commit |

### Tier 2: Execute (requires explicit approval)

| Tool | Description |
|------|-------------|
| `exec.run` | Run allowlisted command |

## Allowlisted Commands

### Unix (macOS/Linux)

| ID | Command |
|----|---------|
| `pytest` | `python -m pytest` |
| `npm_test` | `npm test` |
| `go_test` | `go test ./...` |
| `cargo_test` | `cargo test` |
| `ruff` | `ruff check` |
| `eslint` | `npx eslint` |
| `prettier` | `npx prettier --write` |
| `black` | `black` |

### Windows (PowerShell)

| ID | Command |
|----|---------|
| `pytest` | `python -m pytest` |
| `npm_test` | `npm test` |
| `dotnet_test` | `dotnet test` |
| `eslint` | `npx eslint` |
| `prettier` | `npx prettier --write` |

## Security

### Path Validation

- All paths are resolved to absolute
- Paths must be within workspace root
- Symlinks escaping workspace are blocked
- Sensitive paths (.env, *.pem, *.key) are blocked by default

### Command Execution

- Only allowlisted commands can run
- No arbitrary shell execution
- Timeouts enforced
- Output size limits

### Secret Redaction

Tool outputs are scanned for common secret patterns and redacted before being sent to the LLM.

## Development

```bash
# Run tests
go test ./...

# Build for all platforms
GOOS=darwin GOARCH=amd64 go build -o bin/tldw-agent-host-darwin-amd64 ./cmd/tldw-agent-host
GOOS=darwin GOARCH=arm64 go build -o bin/tldw-agent-host-darwin-arm64 ./cmd/tldw-agent-host
GOOS=linux GOARCH=amd64 go build -o bin/tldw-agent-host-linux-amd64 ./cmd/tldw-agent-host
GOOS=windows GOARCH=amd64 go build -o bin/tldw-agent-host-windows-amd64.exe ./cmd/tldw-agent-host
```

## License

MIT
