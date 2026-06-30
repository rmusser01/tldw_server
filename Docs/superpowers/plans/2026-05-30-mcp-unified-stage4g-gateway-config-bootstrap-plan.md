# MCP Unified Stage 4G Gateway Config Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for behavior changes and keep this slice package-owned.

**Goal:** Add package-owned gateway config bootstrap helpers that construct profile-aware gateway runtimes from explicit memory or SQLite profile-store configuration.

**Architecture:** Keep the config layer under `mcp_unified.gateway` with no `tldw_Server_API` imports. Use small dataclasses instead of a broad settings framework. The config helper should resolve the profile store, then delegate to the Stage 4F `bootstrap_profile_gateway()` function so profile seeding, collision behavior, and runtime creation stay centralized.

**Tech Stack:** Python 3.11, dataclasses, package-local gateway/profile/storage primitives, pytest, Ruff, Bandit.

---

### Task 1: Config Bootstrap RED Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
- Create: `mcp_unified/gateway/config.py`
- Modify: `mcp_unified/gateway/__init__.py`

- [x] **Step 1: Add failing config-bootstrap tests**

Add tests that assert:
- a memory-store config can seed a built-in default preset and build a `ProfileAwareGatewayRuntime`;
- a SQLite-store config creates a `SQLiteMCPStore`, seeds the default preset, and persists the profile;
- an injected profile store is preserved even when config contains a different store setting;
- invalid store kinds and missing/blank SQLite paths raise clear `ValueError`s;
- profile mapping inputs are copy-isolated during config construction.

- [x] **Step 2: Run RED tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage4g-gateway-config-bootstrap/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: new tests fail because `mcp_unified.gateway.config` does not exist.

Result: `5 failed, 43 passed, 4 warnings`; all new failures were the expected missing `mcp_unified.gateway.config` module.

### Task 2: Package Config Helper

**Files:**
- Create: `mcp_unified/gateway/config.py`
- Modify: `mcp_unified/gateway/__init__.py`

- [x] **Step 1: Add config dataclasses**

Implement:
- `GatewayProfileStoreConfig(kind="memory" | "sqlite", sqlite_path=None)`;
- `GatewayProfileBootstrapConfig(store=..., profiles=..., default_profile_id=None, default_preset_id=None)`;
- validation that rejects unknown store kinds and SQLite configs without a path.

- [x] **Step 2: Add config bootstrap helper**

Implement `bootstrap_profile_gateway_from_config(backend, config, *, profile_store=None)`:
- validate the config;
- use the injected `profile_store` when supplied;
- otherwise create `InMemoryProfileStore` or `SQLiteMCPStore` from config;
- delegate to `bootstrap_profile_gateway()`;
- return `GatewayProfileBootstrap`.

- [x] **Step 3: Export the public config APIs**

Export the config dataclasses and helper from `mcp_unified.gateway` without importing host code or eagerly importing SQLite.

- [x] **Step 4: Run GREEN tests**

Run the focused gateway package tests and confirm all pass.

Result: `48 passed, 4 warnings`.

- [x] **Step 5: Address review follow-up**

After rebasing onto `origin/dev` `1c91138f5320a623002e4da160966cbfbeab9ead`, verify and address the still-valid review comments:
- reject empty or whitespace-only SQLite profile-store paths;
- copy-isolate profile mapping/model inputs during config construction;
- keep the bootstrap store normalization type-safe;
- add intent docstrings to the Stage 4G config bootstrap tests.

Review RED result: `3 failed, 48 passed, 4 warnings`.
Review GREEN result: `51 passed, 4 warnings`.

### Task 3: Compatibility, Security, And PR Handoff

**Files:**
- Modify: `Docs/superpowers/plans/2026-05-30-mcp-unified-stage4g-gateway-config-bootstrap-plan.md`
- Modify: `backlog/tasks/task-564 - Implement-MCP-Unified-Stage-4G-gateway-config-bootstrap.md`

- [x] **Step 1: Run host compatibility tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage4g-gateway-config-bootstrap/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py -q
```

- [x] **Step 2: Run lint, security, and whitespace checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage4g-gateway-config-bootstrap/.venv/bin/python -m ruff check mcp_unified/gateway tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage4g-gateway-config-bootstrap/.venv/bin/python -m bandit -r mcp_unified/gateway -f json -o /tmp/bandit_mcp_stage4g_gateway_config_bootstrap.json
git diff --check
```

- [x] **Step 3: Update plan and Backlog task**

Record RED/GREEN and final verification evidence. Check off acceptance criteria and Definition of Done.

- [x] **Step 4: Commit, push, and open PR**

Commit the plan, gateway/test changes, and Backlog task update together, push the branch, and open a PR against `dev`.

## Verification

- Baseline: `43 passed, 4 warnings` for `test_gateway_fastapi_package.py` on merged Stage 4F.
- RED: `5 failed, 43 passed, 4 warnings`; expected missing config module failures only.
- GREEN: `48 passed, 4 warnings` for `test_gateway_fastapi_package.py`.
- Review RED: `3 failed, 48 passed, 4 warnings` for blank SQLite path validation and profile input copy isolation.
- Review GREEN: `51 passed, 4 warnings` for `test_gateway_fastapi_package.py`.
- Compatibility: `47 passed, 4 warnings` for `test_extraction_contracts.py` and `test_http_mapping.py`.
- Ruff: `All checks passed!` for `mcp_unified/gateway` and the focused gateway package tests.
- Bandit: `0` results and no errors for `mcp_unified/gateway`.
- Whitespace: `git diff --check` passed.
