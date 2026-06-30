# MCP Unified Stage 4E Gateway Profile Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a package-owned profile-aware gateway runtime wrapper for standalone MCP gateway discovery and tool execution.

**Architecture:** Keep profile enforcement host-neutral and package-owned under `mcp_unified.gateway`. The wrapper resolves the explicit transport-selected profile or configured standalone default profile, derives the effective profile policy through existing profile resolution primitives, filters `tools/list`, and denies `tools/call` before backend delegation when the profile is missing, disabled, unavailable, or does not allow the requested tool/capability. This slice intentionally avoids external MCP lifecycle, upstream process spawning, SQLite CLI/config commands, preset duplication flows, and `tldw_server` host route integration.

**Tech Stack:** Python 3.11, Pydantic profile models, package-local profile resolver/store primitives, FastAPI metadata extraction, pytest, Ruff, Bandit.

---

### Task 1: Profile-Aware Gateway RED Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
- Create: `mcp_unified/gateway/profile_runtime.py`
- Modify: `mcp_unified/gateway/runtime.py`
- Modify: `mcp_unified/gateway/jsonrpc.py`
- Modify: `mcp_unified/gateway/fastapi.py`
- Modify: `mcp_unified/gateway/__init__.py`

- [x] **Step 1: Add fake profile-gated runtime coverage**

Add tests that construct a profile-aware gateway runtime around a fake backend runtime and an in-memory profile store.

Required failing behaviors:
- no default or explicit profile returns an empty `tools/list`;
- no default or explicit profile denies `tools/call` with a structured JSON-RPC policy error carrying `reason_code="profile_required"`;
- a default profile allows the configured tool and filters denied tools from discovery;
- an explicit FastAPI profile header selects the profile when no default is configured;
- denied tools produce a machine-readable policy denial and do not reach the backend runtime.

- [x] **Step 2: Run RED tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: fail because `mcp_unified.gateway.profile_runtime` and policy-error JSON-RPC mapping do not exist yet.

Evidence: baseline gateway package tests passed with `29 passed, 4 warnings`. After adding the Stage 4E tests, RED failed with `4 failed, 29 passed, 4 warnings`; all four new tests failed with `ModuleNotFoundError: No module named 'mcp_unified.gateway.profile_runtime'`.

### Task 2: Profile Runtime Wrapper And Transport Metadata

**Files:**
- Create: `mcp_unified/gateway/profile_runtime.py`
- Modify: `mcp_unified/gateway/runtime.py`
- Modify: `mcp_unified/gateway/jsonrpc.py`
- Modify: `mcp_unified/gateway/fastapi.py`
- Modify: `mcp_unified/gateway/__init__.py`

- [x] **Step 1: Add structured gateway policy denial**

Add a package-local exception for policy denials with reason code, status, provenance, and warnings. Map it in JSON-RPC to a custom server error response so execution denials are machine-readable rather than generic internal errors.

- [x] **Step 2: Add `ProfileAwareGatewayRuntime`**

Implement a wrapper that:
- proxies runtime identity to the backend runtime;
- resolves profiles from existing `StoreBackedProfileResolver` / structured resolution primitives;
- reads explicit profile selection from `GatewayRequestContext.metadata["profile_id"]`;
- returns no executable tools for unresolved profiles;
- filters `tools/list` by effective profile policy;
- denies `tools/call` before backend delegation when profile resolution or tool policy fails;
- delegates resource, prompt, and module methods unchanged in this slice.

- [x] **Step 3: Add lightweight FastAPI profile metadata extraction**

Propagate an optional profile id from `X-MCP-Profile` / `X-MCP-Profile-Id` headers, or `profile_id` / `profileId` query params, into the gateway request metadata for HTTP and WebSocket transports. Existing requests without profile metadata must keep their current behavior for unwrapped runtimes.

- [x] **Step 4: Run GREEN gateway tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: all gateway package tests pass.

Evidence: gateway package tests passed with `33 passed, 4 warnings` after adding `ProfileAwareGatewayRuntime`, structured policy-denial JSON-RPC mapping, and HTTP/WebSocket profile metadata propagation.

### Task 3: Compatibility, Security, And PR Handoff

**Files:**
- Modify: `Docs/superpowers/plans/2026-05-30-mcp-unified-stage4e-gateway-profile-runtime-plan.md`
- Modify: `backlog/tasks/task-562 - Implement-MCP-Unified-Stage-4E-gateway-profile-runtime.md`

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
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/gateway -f json -o /tmp/bandit_mcp_stage4e_gateway_profile_runtime.json
git diff --check
```

Expected: Ruff passes, Bandit reports no findings for `mcp_unified/gateway`, and whitespace check is clean.

Evidence: Ruff reported `All checks passed!`; Bandit JSON at `/tmp/bandit_mcp_stage4e_gateway_profile_runtime.json` reported `0` findings and no errors; `git diff --check` exited cleanly.

- [x] **Step 3: Update plan and Backlog task**

Record RED/GREEN and final verification evidence in this plan and TASK-562. Check off completed acceptance criteria and Definition of Done.

- [x] **Step 4: Commit, push, and open PR**

Commit the plan, gateway/test changes, and Backlog task update together, push the branch, and open a PR against `dev`.
