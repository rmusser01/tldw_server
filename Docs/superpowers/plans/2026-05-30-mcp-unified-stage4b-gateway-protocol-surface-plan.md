# MCP Unified Stage 4B Gateway Protocol Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the standalone gateway skeleton from Stage 4A to cover the package JSON-RPC discovery surface for resources, prompts, and modules through an injected runtime protocol.

**Architecture:** Keep the transport package-owned under `mcp_unified.gateway` and continue to avoid `tldw_Server_API` imports. The FastAPI gateway remains a thin JSON-RPC transport: it validates envelope/params, builds `GatewayRequestContext`, and delegates runtime-owned behavior through `GatewayRuntime`. This slice intentionally avoids SQLite wiring, external MCP lifecycle, client-facing stdio, default-profile enforcement, and host route integration.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v1/v2 compatibility, pytest, Ruff, Bandit.

---

### Task 1: Gateway Protocol Surface RED Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
- Modify: `mcp_unified/gateway/runtime.py`
- Modify: `mcp_unified/gateway/fastapi.py`

- [x] **Step 1: Extend the fake runtime fixture**

Add fake runtime methods and call capture lists for:

```python
async def list_resources(self, context): ...
async def read_resource(self, uri, context): ...
async def list_prompts(self, context): ...
async def get_prompt(self, name, arguments, context): ...
async def list_modules(self, context): ...
async def get_modules_health(self, context): ...
```

- [x] **Step 2: Write failing tests for resource, prompt, and module methods**

Add tests that POST to `/mcp/request` and expect:

```python
{"jsonrpc": "2.0", "method": "resources/list", "params": {}, "id": "resources-1"}
```

returns `result.resources`, and:

```python
{"jsonrpc": "2.0", "method": "resources/read", "params": {"uri": "resource://unit/doc"}, "id": "read-1"}
```

returns `result.contents`.

Add equivalent prompt and module checks for:

- `prompts/list` -> `result.prompts`
- `prompts/get` with `{"name": "review.prompt", "arguments": {"topic": "gateway"}}` -> runtime prompt result
- `modules/list` -> `result.modules`
- `modules/health` -> `result.health`

Assert each runtime call receives the expected `request_id` in its context.

- [x] **Step 3: Write failing validation tests**

Add tests that verify:

- `resources/read` without a non-empty string `uri` returns JSON-RPC `-32602`
- `prompts/get` without a non-empty string `name` returns JSON-RPC `-32602`
- `prompts/get.arguments` as a falsy non-object such as `[]` returns JSON-RPC `-32602` and does not call the runtime

- [x] **Step 4: Run RED tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: the new tests fail with `Method not found` or missing runtime protocol methods while the existing Stage 4A tests remain valid.

Evidence: `4 failed, 11 passed, 3 warnings`; the new resource/prompt/module tests failed with JSON-RPC `Method not found` responses before implementation.

### Task 2: Gateway Runtime Protocol And Dispatch

**Files:**
- Modify: `mcp_unified/gateway/runtime.py`
- Modify: `mcp_unified/gateway/fastapi.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [x] **Step 1: Extend `GatewayRuntime`**

Add protocol methods:

```python
async def list_resources(self, context: GatewayRequestContext) -> list[dict[str, Any]]: ...
async def read_resource(self, uri: str, context: GatewayRequestContext) -> dict[str, Any]: ...
async def list_prompts(self, context: GatewayRequestContext) -> list[dict[str, Any]]: ...
async def get_prompt(self, name: str, arguments: dict[str, Any], context: GatewayRequestContext) -> dict[str, Any]: ...
async def list_modules(self, context: GatewayRequestContext) -> list[dict[str, Any]]: ...
async def get_modules_health(self, context: GatewayRequestContext) -> dict[str, Any]: ...
```

- [x] **Step 2: Add small string validators in `fastapi.py`**

Add a helper that accepts a value and field name and raises `ValueError` unless the value is a non-empty string after stripping.

- [x] **Step 3: Extend `_dispatch_jsonrpc()`**

Add branches:

```python
if method == "resources/list":
    return {"resources": await runtime.list_resources(context)}
if method == "resources/read":
    uri = _required_string(params.get("uri"), "resources/read requires a non-empty string uri")
    return await runtime.read_resource(uri, context)
if method == "prompts/list":
    return {"prompts": await runtime.list_prompts(context)}
if method == "prompts/get":
    name = _required_string(params.get("name"), "prompts/get requires a non-empty string name")
    arguments = _object_or_empty(params.get("arguments"), "prompts/get arguments must be an object")
    return await runtime.get_prompt(name, arguments, context)
if method == "modules/list":
    return {"modules": await runtime.list_modules(context)}
if method == "modules/health":
    return {"health": await runtime.get_modules_health(context)}
```

Do not introduce profile/default policy behavior in this slice.

- [x] **Step 4: Run GREEN tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: all gateway package tests pass.

Evidence: `15 passed, 3 warnings`.

### Task 3: Compatibility, Security, And Task Closeout

**Files:**
- Modify: `Docs/superpowers/plans/2026-05-30-mcp-unified-stage4b-gateway-protocol-surface-plan.md`
- Modify: `backlog/tasks/task-559 - Implement-MCP-Unified-Stage-4B-gateway-protocol-surface.md`

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
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/gateway -f json -o /tmp/bandit_mcp_stage4b_gateway_protocol_surface.json
git diff --check
```

Expected: Ruff passes, Bandit reports no findings for `mcp_unified/gateway`, and whitespace check is clean.

Evidence: Ruff reported `All checks passed!`; Bandit JSON reported `"results": []`; `git diff --check` exited cleanly.

- [x] **Step 3: Update plan and Backlog task**

Record RED/GREEN and final verification evidence in this plan and TASK-559. Check off completed acceptance criteria and Definition of Done.

- [x] **Step 4: Commit and push**

Commit the plan, runtime/transport/test changes, and Backlog task update together.
