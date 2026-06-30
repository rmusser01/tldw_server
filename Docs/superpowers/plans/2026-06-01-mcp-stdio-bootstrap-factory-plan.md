# MCP Stdio Bootstrap Factory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the package-owned stdio external transport factory into standalone gateway config bootstrap.

**Architecture:** Add an explicit opt-in external runtime config nested under `GatewayProfileBootstrapConfig`. When enabled, `bootstrap_profile_gateway_from_config()` creates `GatewayExternalRuntimeManager` from the same external registry storage bundle already used for registry management, using `create_external_transport` unless the caller injects a factory or full runtime manager.

**Tech Stack:** Python dataclasses, async gateway bootstrap helpers, package-owned MCP Unified runtime, pytest.

---

### Task 1: Bootstrap Config Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [x] Add tests proving external runtime is disabled by default for SQLite config.
- [x] Add tests proving enabled SQLite config creates a runtime manager with the package stdio factory.
- [x] Add tests proving injected transport factory overrides the package default.
- [x] Add tests proving unsupported external runtime factory config is rejected.
- [x] Add tests proving memory config cannot enable runtime management without external registry storage.
- [x] Run the new tests and confirm they fail for missing config/runtime wiring.

### Task 2: Config Runtime Wiring

**Files:**
- Modify: `mcp_unified/gateway/config.py`
- Modify: `mcp_unified/gateway/__init__.py`

- [x] Add `GatewayExternalRuntimeBootstrapConfig`.
- [x] Normalize `external_runtime` in `GatewayProfileBootstrapConfig.__post_init__`.
- [x] Add optional runtime dependency injection parameters to `bootstrap_profile_gateway_from_config()`.
- [x] Build `GatewayExternalRuntimeManager` only when runtime config is enabled and no manager is injected.
- [x] Use `mcp_unified.federation.create_external_transport` as the default factory.
- [x] Export the new config type from package gateway modules.
- [x] Run the focused tests and confirm they pass.

### Task 3: Validation

**Files:**
- Update: `backlog/tasks/task-583 - Wire-MCP-stdio-transport-factory-into-gateway-bootstrap.md`

- [x] Run focused pytest for gateway config/bootstrap tests.
- [x] Run Ruff on touched Python files.
- [x] Run Bandit on touched Python source.
- [x] Run `git diff --check`.
- [x] Update Backlog task notes and verification.
- [x] Commit, push, and open the PR.
