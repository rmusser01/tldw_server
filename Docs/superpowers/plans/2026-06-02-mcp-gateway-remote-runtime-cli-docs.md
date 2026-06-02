# MCP Gateway Remote Runtime CLI And Docs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CLI commands that control external runtime lifecycle operations by calling a running standalone gateway, then document the end-to-end standalone admin workflow.

**Architecture:** Keep process ownership clear: local CLI store commands edit persistent config, while remote runtime CLI commands call HTTP admin endpoints on an already-running gateway. Use a small package-owned remote admin client with stdlib HTTP transport unless the package already depends on an HTTP client; pass admin auth through headers from environment/config without logging secret values.

**Tech Stack:** Python, argparse, stdlib `urllib.request` or existing HTTP client if already required, FastAPI admin routes, pytest, Bandit, Markdown docs.

**Backlog Task:** `TASK-593`

**Depends On:** `TASK-591`, `TASK-592`

---

## File Structure

- Create `mcp_unified/gateway/remote_admin.py`: remote gateway admin client, URL normalization, auth header handling, JSON request/response helpers, sanitized errors.
- Modify `mcp_unified/gateway/cli.py`: add remote runtime commands.
- Modify `mcp_unified/gateway/fastapi.py`: only if route response shapes need minor normalization discovered by tests.
- Create `Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md`: standalone gateway admin/config usage guide.
- Create `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_remote_runtime_cli.py`: remote client and CLI command behavior.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`: parser/help coverage if existing CLI tests centralize command assertions.

## Acceptance Criteria

- CLI can call a running gateway for runtime list/start/stop/restart/refresh/reconcile/install/update operations.
- CLI runtime commands require an explicit `--gateway-url` or `MCP_UNIFIED_GATEWAY_URL`.
- Admin auth uses an environment-provided value such as `MCP_UNIFIED_GATEWAY_ADMIN_KEY`; command-line secret arguments are avoided.
- CLI preserves the gateway JSON payloads and reason codes.
- Docs explain local store commands versus remote runtime commands and include a safe credential-grant example.
- No runtime CLI command starts an upstream process that becomes orphaned when the CLI exits.

### Task 1: Add Remote Client Failing Tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_remote_runtime_cli.py`

- [ ] **Step 1: Write URL normalization tests**

Assert base URLs with or without trailing slash resolve paths such as `/mcp/external-servers/runtime` correctly.

- [ ] **Step 2: Write auth header tests**

Assert the remote client sends the configured admin header when `MCP_UNIFIED_GATEWAY_ADMIN_KEY` is set and omits it when absent.

- [ ] **Step 3: Write response/error tests**

Use a fake request function to assert JSON payload passthrough for success and sanitized error envelopes for malformed responses or connection failures.

- [ ] **Step 4: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_remote_runtime_cli.py -v
```

Expected: fail because `mcp_unified.gateway.remote_admin` does not exist.

### Task 2: Implement Remote Admin Client

**Files:**
- Create: `mcp_unified/gateway/remote_admin.py`

- [ ] **Step 1: Add client config**

Create a small dataclass:

```python
@dataclass(frozen=True, slots=True)
class RemoteGatewayAdminConfig:
    gateway_url: str
    admin_header_name: str = "X-MCP-Gateway-Admin-Key"
    admin_key: str | None = None
    timeout_seconds: float = 30.0
```

- [ ] **Step 2: Add request helper**

Implement JSON GET/POST helpers. Prefer stdlib `urllib.request` to avoid a new runtime dependency unless the package already requires an HTTP client.

- [ ] **Step 3: Add runtime methods**

Implement:

```python
list_runtime_servers()
start_server(server_id)
stop_server(server_id)
restart_server(server_id)
refresh_server(server_id: str | None = None)
reconcile(server_id: str | None = None)
install_server(server_id)
update_server(server_id)
```

- [ ] **Step 4: Run client tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_remote_runtime_cli.py -v
```

Expected: pass for client-level tests.

### Task 3: Add Remote Runtime CLI Commands

**Files:**
- Modify: `mcp_unified/gateway/cli.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_remote_runtime_cli.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py` if needed.

- [ ] **Step 1: Write CLI command tests**

Cover flat commands to match existing CLI style:

- `runtime-list`
- `runtime-start <server_id>`
- `runtime-stop <server_id>`
- `runtime-restart <server_id>`
- `runtime-refresh [server_id]`
- `runtime-reconcile [server_id]`
- `runtime-install <server_id>`
- `runtime-update <server_id>`

Each command should accept `--gateway-url`; tests should also cover `MCP_UNIFIED_GATEWAY_URL`.

- [ ] **Step 2: Add common remote options**

Add helpers for:

- `--gateway-url`
- `--admin-header-name`
- admin key from `MCP_UNIFIED_GATEWAY_ADMIN_KEY`
- timeout from optional `--timeout-seconds`

Avoid `--admin-key` to keep secrets out of process lists.

- [ ] **Step 3: Implement handlers**

Call `RemoteGatewayAdminClient` methods and emit exactly one JSON object to stdout or stderr, matching existing CLI behavior.

- [ ] **Step 4: Run CLI tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_remote_runtime_cli.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  -v
```

Expected: pass.

### Task 4: Add Standalone Admin Documentation

**Files:**
- Create: `Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md`

- [ ] **Step 1: Document concepts**

Explain:

- profile store versus running gateway runtime.
- local config CLI commands versus remote runtime CLI commands.
- credential grants as broker metadata, not secret storage.
- admin auth setup and expected header/env vars.

- [ ] **Step 2: Add safe examples**

Include examples for:

- validating config.
- duplicating a preset.
- creating an external server.
- creating a credential grant with broker id and slot only.
- exporting/importing a snapshot.
- calling `runtime-list`, `runtime-start`, and `runtime-refresh`.

- [ ] **Step 3: Add operational cautions**

State that runtime commands require a running gateway and do not start durable upstream processes from a short-lived local CLI.

### Task 5: Final Verification

**Files:**
- Modified files from prior tasks.
- Backlog task `TASK-593`.

- [ ] **Step 1: Run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_remote_runtime_cli.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  -v
```

- [ ] **Step 2: Run Bandit**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  mcp_unified/gateway/remote_admin.py \
  mcp_unified/gateway/cli.py \
  mcp_unified/gateway/fastapi.py \
  -f json -o /tmp/bandit_mcp_gateway_remote_runtime_cli.json
```

- [ ] **Step 3: Update Backlog**

Record touched files, verification commands, known skips, and final summary in `TASK-593`.
