# MCP Unified Stage 4A Gateway Entrypoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the smallest package-owned FastAPI gateway skeleton for standalone MCP use.

**Architecture:** Keep runtime code inside `mcp_unified.gateway` so it can be imported without `tldw_Server_API`. Keep tests in the existing host MCP test suite so they are not packaged with the standalone runtime. The gateway exposes an app/router factory that accepts an injected runtime protocol and handles status, `initialize`, `ping`, `tools/list`, and `tools/call` JSON-RPC requests for fake or future standalone runtimes.

**Tech Stack:** Python 3.11, FastAPI, pytest, package-local `mcp_unified` interfaces.

---

### Task 1: Gateway Skeleton Contract Tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
- Create: `mcp_unified/gateway/__init__.py`
- Create: `mcp_unified/gateway/runtime.py`
- Create: `mcp_unified/gateway/fastapi.py`

- [x] **Step 1: Write the failing tests**

Add tests that:
- AST-scan `mcp_unified/gateway/*.py` and fail on any `tldw_Server_API` import.
- Build a minimal app with `create_gateway_app(fake_runtime)`.
- Assert `GET /mcp/status` returns gateway metadata.
- Assert `POST /mcp/request` handles `initialize`, `tools/list`, and `tools/call`.

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: fail because `mcp_unified.gateway` does not exist.

- [x] **Step 3: Implement the minimal package gateway**

Create:
- `GatewayRequestContext` dataclass with request id, client id, user id, metadata.
- `GatewayRuntime` protocol with `name`, `version`, `list_tools()`, and `call_tool()`.
- `create_gateway_router(runtime)` and `create_gateway_app(runtime, prefix="/mcp")`.

Keep implementation deliberately small:
- No SQLite store wiring.
- No client-facing stdio.
- No upstream external stdio process lifecycle.
- No host `MCPServer` or `MCPProtocol` import.

- [x] **Step 4: Run focused tests to verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: pass.

### Task 2: Host Compatibility And Validation

**Files:**
- Modify: `backlog/tasks/task-557 - Implement-MCP-Unified-Stage-4A-gateway-entrypoint-skeleton.md`

- [x] **Step 1: Run focused host compatibility tests**

Run:

```bash
.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py -q
```

Expected: existing host extraction and HTTP behavior remains compatible.

- [x] **Step 2: Run lint, security, and whitespace checks**

Run:

```bash
.venv/bin/python -m ruff check mcp_unified/gateway tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
.venv/bin/python -m bandit -r mcp_unified/gateway -f json -o /tmp/bandit_mcp_stage4a_gateway.json
git diff --check
```

Expected: Ruff passes, Bandit reports 0 findings, whitespace check is clean.

- [x] **Step 3: Record Backlog verification and commit**

Update TASK-557 with implementation notes, verification results, known skips, and final summary. Commit the plan, gateway code, tests, and task update together.

**Verification:**
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest mcp_unified/tests/test_gateway_fastapi.py -q` -> RED failed with `ModuleNotFoundError: No module named 'mcp_unified.gateway'`.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q` -> 2 passed, 3 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py -q` -> 47 passed, 4 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check mcp_unified/gateway tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py` -> All checks passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/gateway -f json -o /tmp/bandit_mcp_stage4a_gateway.json` -> 0 findings.
- `git diff --check` -> clean.
