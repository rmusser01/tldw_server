# MCP Unified Stage 4D Gateway Stdio Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a package-owned client-facing stdio JSON-RPC transport skeleton for standalone MCP gateway runtimes.

**Architecture:** Keep stdio host-neutral and package-owned under `mcp_unified.gateway`. Extract the existing JSON-RPC request/response parsing, validation, dispatch, notification suppression, and response serialization into a transport-neutral gateway module. FastAPI HTTP/WebSocket and stdio should share that dispatcher. This slice intentionally avoids SQLite/profile enforcement, external MCP lifecycle, upstream stdio process spawning, auth/session policy, and host route integration.

**Tech Stack:** Python 3.11, asyncio streams, Pydantic v1/v2 compatibility, pytest, Ruff, Bandit.

---

### Task 1: Gateway Stdio RED Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
- Create: `mcp_unified/gateway/jsonrpc.py`
- Create: `mcp_unified/gateway/stdio.py`
- Modify: `mcp_unified/gateway/fastapi.py`
- Modify: `mcp_unified/gateway/__init__.py`

- [x] **Step 1: Add stdio import-boundary coverage**

Extend the existing gateway package import scan so all `mcp_unified/gateway/*.py` files, including `stdio.py` and `jsonrpc.py`, fail if they import `tldw_Server_API`.

- [x] **Step 2: Write failing stdio line-handler tests**

Add focused tests that call a package stdio handler with a fake runtime and assert:

- a JSON-RPC `initialize` input line returns one newline-terminated JSON-RPC response line;
- a `ping` notification input returns no output;
- a mixed batch input returns one JSON array response line with notification responses omitted;
- invalid JSON returns a JSON-RPC `-32700` parse error response line;
- runtime contexts carry `metadata["path"] == "stdio://stdin"` and `metadata["transport"] == "stdio"`.

- [x] **Step 3: Run RED tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: fail because `mcp_unified.gateway.stdio` does not exist yet.

Evidence: baseline gateway package tests passed with `22 passed, 3 warnings`; after adding stdio tests, RED failed with `4 failed, 22 passed, 3 warnings` because `mcp_unified.gateway.stdio` did not exist.

### Task 2: Shared Gateway JSON-RPC Core And Stdio Transport

**Files:**
- Create: `mcp_unified/gateway/jsonrpc.py`
- Create: `mcp_unified/gateway/stdio.py`
- Modify: `mcp_unified/gateway/fastapi.py`
- Modify: `mcp_unified/gateway/__init__.py`

- [x] **Step 1: Extract transport-neutral JSON-RPC helpers**

Move the gateway request/response models, parsing helpers, dispatcher, error mapping, and response serialization into `mcp_unified.gateway.jsonrpc`. Replace FastAPI `Response(status_code=204)` as the internal notification sentinel with a package-local no-response sentinel.

- [x] **Step 2: Keep FastAPI behavior stable**

Update `mcp_unified.gateway.fastapi` so HTTP still returns 204 for notification-only requests and WebSocket still suppresses notification-only responses. Keep initialize, resources, prompts, modules, validation, batch, binary-frame, and disconnect behavior unchanged.

- [x] **Step 3: Add stdio transport helpers**

Add `GatewayStdioServer` and `handle_stdio_line(...)` that:

- accept line-delimited JSON-RPC payloads from stdin-style text;
- return `str | None`, where `None` means notification-only/no response;
- append exactly one trailing newline for returned response lines;
- serialize single and batch JSON-RPC responses with the shared JSON-safe serializer;
- build a `GatewayRequestContext` with `path="stdio://stdin"` and `transport="stdio"`.

- [x] **Step 4: Run GREEN gateway tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: all gateway package tests pass.

Evidence: gateway package tests passed with `27 passed, 4 warnings` after adding `mcp_unified.gateway.jsonrpc`, `mcp_unified.gateway.stdio`, lazy FastAPI helper exports, and FastAPI shared-dispatcher wiring.

### Task 3: Compatibility, Security, And PR Handoff

**Files:**
- Modify: `Docs/superpowers/plans/2026-05-30-mcp-unified-stage4d-gateway-stdio-transport-plan.md`
- Modify: `backlog/tasks/task-561 - Implement-MCP-Unified-Stage-4D-gateway-stdio-transport.md`

- [x] **Step 1: Run host compatibility tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py -q
```

Expected: existing host extraction and HTTP mapping tests pass.

Evidence: host compatibility tests passed with `47 passed, 4 warnings`.

- [x] **Step 2: Run lint, security, and whitespace checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check mcp_unified/gateway tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/gateway -f json -o /tmp/bandit_mcp_stage4d_gateway_stdio_transport.json
git diff --check
```

Expected: Ruff passes, Bandit reports no findings for `mcp_unified/gateway`, and whitespace check is clean.

Evidence: initial Ruff found two import-order issues and `ruff check --fix` fixed both. Final Ruff reported `All checks passed!`; Bandit JSON at `/tmp/bandit_mcp_stage4d_gateway_stdio_transport.json` reported `0` results and no errors; `git diff --check` exited cleanly.

- [x] **Step 3: Update plan and Backlog task**

Record RED/GREEN and final verification evidence in this plan and TASK-561. Check off completed acceptance criteria and Definition of Done.

- [x] **Step 4: Commit, push, and open PR**

Commit the plan, gateway/test changes, and Backlog task update together, push the branch, and open a PR against `dev`.
