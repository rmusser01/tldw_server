# MCP Unified Stage 4C Gateway WebSocket Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add package-owned FastAPI WebSocket JSON-RPC transport for the standalone MCP gateway.

**Architecture:** Keep the package transport thin and host-neutral. The WebSocket endpoint accepts JSON text/object frames, reuses the existing JSON-RPC validation/dispatch/error mapping, sends only JSON-RPC response envelopes, and suppresses notification-only responses the same way HTTP returns 204. This slice does not add auth/session policy, stdio, SQLite wiring, external lifecycle, or host route integration.

**Tech Stack:** Python 3.11, FastAPI/TestClient WebSocket support, Pydantic v1/v2 compatibility, pytest, Ruff, Bandit.

---

### Task 1: Gateway WebSocket RED Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
- Modify: `mcp_unified/gateway/fastapi.py`

- [x] **Step 1: Add WebSocket success coverage**

Add a package gateway test that connects to `/mcp/ws`, sends `initialize`, `ping`, and one Stage 4B discovery request such as `resources/list`, and asserts:

- every response has `jsonrpc == "2.0"`
- response ids are echoed
- payloads match the HTTP transport results
- runtime context request ids are propagated

- [x] **Step 2: Add WebSocket parse-error coverage**

Add a test that sends invalid JSON text over `/mcp/ws` and expects a JSON-RPC `-32700` parse error response with `id: null`.

- [x] **Step 3: Add notification and batch coverage**

Add a test that sends a notification-only `ping` frame and then a regular `ping` request on the same WebSocket, proving the notification produces no response and the next request still succeeds.

Add a test that sends a mixed batch with one notification and one request and asserts only the request response is sent.

- [x] **Step 4: Run RED tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: the new WebSocket tests fail because `/mcp/ws` is not exposed yet.

Evidence: `4 failed, 16 passed, 3 warnings`; all new WebSocket tests failed at connection setup because `/mcp/ws` was not registered.

### Task 2: Package WebSocket Transport

**Files:**
- Modify: `mcp_unified/gateway/fastapi.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [x] **Step 1: Import FastAPI WebSocket primitives**

Add `WebSocket` and `WebSocketDisconnect` imports from FastAPI.

- [x] **Step 2: Add raw payload parsing helper**

Extract raw JSON parsing into a helper that accepts a text/bytes frame and returns the same JSON-RPC parse error object used by HTTP for malformed JSON.

- [x] **Step 3: Add WebSocket send helper**

Add a helper that sends JSON-RPC response models/lists through `websocket.send_json(...)` and intentionally does nothing for notification-only `Response(status_code=204)` results.

- [x] **Step 4: Add `/ws` route**

Inside `create_gateway_router(runtime)`, add:

```python
@router.websocket("/ws")
async def gateway_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    while True:
        frame = await websocket.receive_text()
        payload = _parse_json_payload(frame)
        if isinstance(payload, _GATEWAY_RESPONSE_TYPES):
            await websocket.send_json(_response_to_json(payload))
            continue
        response = await _handle_jsonrpc(runtime, payload, websocket)
        await _send_websocket_response(websocket, response)
```

Use a request-context metadata path compatible with WebSocket objects.

- [x] **Step 5: Run GREEN tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: all gateway package tests pass.

Evidence: `20 passed, 3 warnings`.

### Task 3: Compatibility, Security, And PR Handoff

**Files:**
- Modify: `Docs/superpowers/plans/2026-05-30-mcp-unified-stage4c-gateway-websocket-transport-plan.md`
- Modify: `backlog/tasks/task-560 - Implement-MCP-Unified-Stage-4C-gateway-WebSocket-transport.md`

- [x] **Step 1: Run host compatibility tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py -q
```

Expected: existing host extraction and HTTP mapping tests pass.

Evidence: `47 passed, 4 warnings`.

- [x] **Step 2: Run lint, security, and whitespace checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check mcp_unified/gateway tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/gateway -f json -o /tmp/bandit_mcp_stage4c_gateway_websocket_transport.json
git diff --check
```

Expected: Ruff passes, Bandit reports no findings for `mcp_unified/gateway`, and whitespace check is clean.

Evidence: Ruff reported `All checks passed!`; Bandit JSON reported `"results": []`; `git diff --check` exited cleanly.

- [x] **Step 3: Update plan and Backlog task**

Record RED/GREEN and final verification evidence in this plan and TASK-560. Check off completed acceptance criteria and Definition of Done.

- [x] **Step 4: Commit, push, and open PR**

Commit the plan, gateway/test changes, and Backlog task update together, push the branch, and open a PR against `dev`.
