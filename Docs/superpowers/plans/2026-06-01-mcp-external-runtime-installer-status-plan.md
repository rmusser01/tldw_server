# MCP External Runtime Installer Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stable, sanitized installer status and install/update operation contracts for the standalone MCP gateway external runtime.

**Architecture:** Keep `GatewayExternalRuntimeManager` as the public normalization boundary around pluggable `ExternalServerInstaller` adapters. Add best-effort installer status collection to runtime status rows and deterministic failure handling for install/update adapter exceptions. Do not implement real third-party package installation.

**Tech Stack:** Python, async/await, FastAPI response models, pytest.

---

### Task 1: Runtime Manager Installer Status Contract

**Files:**
- Modify: `mcp_unified/gateway/external_runtime.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py`

- [ ] **Step 1: Write failing runtime-manager tests**

Add tests that:
- `list_runtime_servers()` includes default unavailable installer status.
- A fake installer status payload is included after sanitization.
- A failing `get_status()` logs diagnostics and returns deterministic unavailable status for that row.

- [ ] **Step 2: Run tests to verify RED**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py -k "installer_status" -q`

Expected: FAIL because runtime rows do not include `installer` status yet.

- [ ] **Step 3: Implement minimal status support**

Add a private manager helper that calls `self._installer.get_status(server.model_copy(deep=True))` with a bounded timeout, sanitizes the payload, applies defaults, catches unexpected exceptions, logs only sanitized diagnostic fields, and returns a stable unavailable status.

- [ ] **Step 4: Run tests to verify GREEN**

Run the same pytest command and confirm it passes.

### Task 2: Install/Update Payload Normalization And Failure Handling

**Files:**
- Modify: `mcp_unified/gateway/external_runtime.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py`

- [ ] **Step 1: Write failing operation tests**

Add tests that:
- Successful fake install/update payloads preserve safe public metadata and apply default fields.
- Nested secret-looking values are removed from operation payloads.
- Adapter exceptions are logged without traceback/raw exception text and raised as `GatewayExternalRuntimeError` with install/update-specific reason codes.

- [ ] **Step 2: Run tests to verify RED**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py -k "installer" -q`

Expected: FAIL because operation payload redaction and adapter exception wrapping are missing.

- [ ] **Step 3: Implement minimal operation normalization**

Centralize installer operation calls, reuse the sanitizer, keep default no-op behavior unchanged, and avoid raw exception text in public errors or logs.

- [ ] **Step 4: Run tests to verify GREEN**

Run the same pytest command and confirm it passes.

### Task 3: FastAPI Contract Coverage

**Files:**
- Modify: `mcp_unified/gateway/fastapi.py` if response models need explicit fields.
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [ ] **Step 1: Write or adjust failing HTTP tests**

Verify runtime status JSON exposes the nested installer object and install/update failure responses use existing external runtime error translation.

- [ ] **Step 2: Run tests to verify RED or baseline pass**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -k "external_runtime" -q`

Expected: FAIL only if FastAPI models block or omit the new response shape; otherwise document that no FastAPI code change is needed.

- [ ] **Step 3: Implement minimal FastAPI model adjustments if needed**

Keep route behavior unchanged and only widen response models where required.

- [ ] **Step 4: Run tests to verify GREEN**

Run the same pytest command and confirm it passes.

### Task 4: Final Verification And Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-585 - Harden-MCP-external-runtime-installer-status-contracts.md`

- [ ] **Step 1: Run focused tests**

Run:
- `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q`
- `source .venv/bin/activate && python -m bandit -r mcp_unified/gateway/external_runtime.py mcp_unified/gateway/fastapi.py -f json -o /tmp/bandit_mcp_external_runtime_installer_status.json`
- `git diff --check`

- [ ] **Step 2: Update Backlog task**

Record touched files, verification results, skipped scope, and final summary in `TASK-585`.

- [ ] **Step 3: Commit**

Commit the implementation, tests, docs, and Backlog task together with a message referencing `TASK-585`.
