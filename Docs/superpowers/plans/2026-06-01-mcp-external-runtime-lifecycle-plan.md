# MCP External Runtime Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe-default startup and shutdown lifecycle handling for standalone gateway external runtimes.

**Architecture:** Introduce a small lifecycle config shared by config bootstrap and FastAPI app creation. FastAPI lifespan delegates startup to `GatewayExternalRuntimeManager.reconcile()` and shutdown to a new `GatewayExternalRuntimeManager.stop_all()` helper only when explicitly enabled.

**Tech Stack:** Python dataclasses, FastAPI lifespan, async gateway runtime manager, pytest.

---

### Task 1: Lifecycle Config And App Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [x] Add a failing test proving default app creation does not call external runtime lifecycle hooks.
- [x] Add a failing test proving opt-in startup reconciliation calls `reconcile()` and records `app.state.external_runtime_startup`.
- [x] Add a failing test proving startup reconcile failure payloads do not block `/status`.
- [x] Add a failing test proving opt-in shutdown calls `stop_all()` and records `app.state.external_runtime_shutdown`.
- [x] Add a failing test proving config bootstrap lifecycle settings flow into `create_gateway_app()`.
- [x] Run the focused tests and confirm they fail for missing lifecycle wiring.

### Task 2: Runtime Shutdown Helper

**Files:**
- Modify: `mcp_unified/gateway/external_runtime.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py`

- [x] Add a failing runtime-manager test for `stop_all()` stopping active servers.
- [x] Implement `GatewayExternalRuntimeManager.stop_all()` using an active-id snapshot under the runtime lock.
- [x] Return a deterministic operation payload with `reason_code="external_runtime_stopped"`, counts, and per-server errors.
- [x] Run the runtime-manager test and confirm it passes.

### Task 3: Config And FastAPI Wiring

**Files:**
- Create: `mcp_unified/gateway/lifecycle.py`
- Modify: `mcp_unified/gateway/bootstrap.py`
- Modify: `mcp_unified/gateway/config.py`
- Modify: `mcp_unified/gateway/fastapi.py`
- Modify: `mcp_unified/gateway/__init__.py`
- Modify: `mcp_unified/gateway/cli.py`

- [x] Add `GatewayExternalRuntimeLifecycleConfig` with boolean validation.
- [x] Carry lifecycle config through `GatewayProfileBootstrap`.
- [x] Extend `GatewayExternalRuntimeBootstrapConfig` with `reconcile_on_startup` and `stop_on_shutdown`.
- [x] Resolve explicit app lifecycle config before bootstrap-carried config.
- [x] Add a FastAPI lifespan hook that performs opt-in startup reconcile and shutdown stop.
- [x] Update CLI validated-config payloads for the new safe-default fields.
- [x] Run the focused FastAPI/config tests and confirm they pass.

### Task 4: Validation And PR

**Files:**
- Update: `backlog/tasks/task-584 - Wire-MCP-external-runtime-startup-and-shutdown-lifecycle.md`

- [x] Run focused pytest for gateway FastAPI/config/runtime tests.
- [x] Run Ruff on touched Python files.
- [x] Run Bandit on touched Python source.
- [x] Run `git diff --check`.
- [x] Update Backlog task notes and verification.
- [x] Commit, push, and open the PR.
