# MCP Gateway Remote Runtime CLI And Docs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CLI commands that control external runtime lifecycle operations by calling a running standalone gateway, then document the end-to-end standalone admin workflow.

**Architecture:** Keep process ownership clear: local CLI store commands edit persistent config, while remote runtime CLI commands call HTTP admin endpoints on an already-running gateway. `--gateway-url` is the gateway admin base URL, including whatever prefix the server mounted, for example `http://127.0.0.1:8000/mcp`; the client only trims trailing slashes and appends endpoint paths. Use a small package-owned remote admin client with stdlib HTTP transport unless the package already depends on an HTTP client; pass admin auth through headers from environment/config without logging secret values.

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
- CLI runtime commands require an explicit `--gateway-url` or `MCP_UNIFIED_GATEWAY_URL`, and that URL is documented as the already-mounted gateway base path such as `http://host/mcp`.
- Admin auth uses an environment-provided value such as `MCP_UNIFIED_GATEWAY_ADMIN_KEY`; command-line secret arguments are avoided.
- CLI preserves the gateway JSON payloads and reason codes for both successful responses and HTTP 4xx/5xx error bodies.
- Docs explain local store commands versus remote runtime commands and include a safe credential-grant example.
- No runtime CLI command starts an upstream process that becomes orphaned when the CLI exits.

### Task 1: Add Remote Client Failing Tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_remote_runtime_cli.py`

- [x] **Step 1: Write URL normalization tests**

Assert base URLs with or without trailing slash resolve endpoint paths correctly. Treat `http://host/mcp` as the gateway base and resolve runtime list to `http://host/mcp/external-servers/runtime`; do not automatically add `/mcp` to `http://host` because gateway prefixes are configurable.

- [x] **Step 2: Write auth header tests**

Assert the remote client sends the configured admin header when `MCP_UNIFIED_GATEWAY_ADMIN_KEY` is set and omits it when absent.

- [x] **Step 3: Write response/error tests**

Use a fake request function to assert JSON payload passthrough for success and sanitized error envelopes for malformed responses or connection failures.

- [x] **Step 4: Write HTTP error preservation tests**

Simulate stdlib `urllib.error.HTTPError` for 401, 404, and 503 responses with JSON bodies. Assert the CLI/client preserves gateway `reason_code`, `server_id`, and public `error` fields instead of replacing them with a generic connection error.

- [x] **Step 5: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_remote_runtime_cli.py -v
```

Expected: fail because `mcp_unified.gateway.remote_admin` does not exist.

### Task 2: Implement Remote Admin Client

**Files:**
- Create: `mcp_unified/gateway/remote_admin.py`

- [x] **Step 1: Add client config**

Create a small dataclass:

```python
@dataclass(frozen=True, slots=True)
class RemoteGatewayAdminConfig:
    gateway_url: str
    admin_header_name: str = "X-MCP-Gateway-Admin-Key"
    admin_key: str | None = None
    timeout_seconds: float = 30.0
```

Validate `gateway_url` is non-blank and has `http` or `https` scheme. Store the normalized base URL without a trailing slash; do not mutate path prefixes beyond trimming trailing slash.

- [x] **Step 2: Add request helper**

Implement JSON GET/POST helpers. Prefer stdlib `urllib.request` to avoid a new runtime dependency unless the package already requires an HTTP client. When `urllib` raises `HTTPError`, read and parse the response body; if it is a JSON object, preserve that payload and exit non-zero from CLI handlers.

- [x] **Step 3: Add runtime methods**

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

- [x] **Step 4: Run client tests**

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

- [x] **Step 1: Write CLI command tests**

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

- [x] **Step 2: Add common remote options**

Add helpers for:

- `--gateway-url`
- `--admin-header-name`
- admin key from `MCP_UNIFIED_GATEWAY_ADMIN_KEY`
- timeout from optional `--timeout-seconds`

Avoid `--admin-key` to keep secrets out of process lists.

Document and test that `--gateway-url` should include the gateway prefix. Example: `--gateway-url http://127.0.0.1:8000/mcp`.

- [x] **Step 3: Implement handlers**

Call `RemoteGatewayAdminClient` methods and emit exactly one JSON object to stdout or stderr, matching existing CLI behavior.

- [x] **Step 4: Run CLI tests**

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

- [x] **Step 1: Document concepts**

Explain:

- profile store versus running gateway runtime.
- local config CLI commands versus remote runtime CLI commands.
- credential grants as broker metadata, not secret storage.
- admin auth setup and expected header/env vars.
- why `--gateway-url` includes the mounted prefix and how that differs from a server origin URL.

- [x] **Step 2: Add safe examples**

Include examples for:

- validating config.
- duplicating a preset.
- creating an external server.
- creating a credential grant with broker id and slot only.
- exporting/importing a snapshot.
- calling `runtime-list`, `runtime-start`, and `runtime-refresh`.
- an HTTP error example showing the gateway `reason_code` preserved in CLI output.

- [x] **Step 3: Add operational cautions**

State that runtime commands require a running gateway and do not start durable upstream processes from a short-lived local CLI.

### Task 5: Final Verification

**Files:**
- Modified files from prior tasks.
- Backlog task `TASK-593`.

- [x] **Step 1: Run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_remote_runtime_cli.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  -v
```

- [x] **Step 2: Run Bandit**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  mcp_unified/gateway/remote_admin.py \
  mcp_unified/gateway/cli.py \
  mcp_unified/gateway/fastapi.py \
  -f json -o /tmp/bandit_mcp_gateway_remote_runtime_cli.json
```

- [x] **Step 3: Update Backlog**

Record touched files, verification commands, known skips, and final summary in `TASK-593`.
