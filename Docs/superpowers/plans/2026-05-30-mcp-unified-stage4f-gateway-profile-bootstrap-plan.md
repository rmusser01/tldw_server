# MCP Unified Stage 4F Gateway Profile Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for behavior changes and keep this slice package-owned.

**Goal:** Add package-owned helpers that bootstrap a profile-aware standalone gateway runtime from built-in presets or caller-supplied profiles.

**Architecture:** Keep bootstrap code under `mcp_unified.gateway` with no `tldw_Server_API` imports. The helper should accept an injected backend `GatewayRuntime`, seed an in-memory or caller-provided `ProfileStore`, select a default profile id, and return `ProfileAwareGatewayRuntime`. This slice intentionally avoids SQLite CLI/config commands, external MCP lifecycle, upstream process spawning, preset editing flows, and host route integration.

**Tech Stack:** Python 3.11, package-local gateway/profile/store primitives, pytest, Ruff, Bandit.

---

### Task 1: Bootstrap Contract RED Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
- Create: `mcp_unified/gateway/bootstrap.py`
- Modify: `mcp_unified/gateway/__init__.py`

- [x] **Step 1: Add import-boundary coverage**

The existing gateway package import scan must include the new bootstrap module and continue to reject any `tldw_Server_API` imports.

- [x] **Step 2: Write failing bootstrap tests**

Add tests that assert:
- `build_profile_gateway_runtime(backend, default_preset_id="project-researcher")` seeds a duplicated default profile and allows a tool advertising `code_search`.
- Caller-supplied profiles can be combined with a default profile id without duplicating built-in presets.
- Unknown default preset ids fail fast with `ValueError`.

- [x] **Step 3: Run RED tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage4f-gateway-profile-bootstrap/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: new bootstrap tests fail because `mcp_unified.gateway.bootstrap` does not exist.

Result: `3 failed, 38 passed, 4 warnings`; all new failures were the expected missing `mcp_unified.gateway.bootstrap` module.

### Task 2: Package Bootstrap Helper

**Files:**
- Create: `mcp_unified/gateway/bootstrap.py`
- Modify: `mcp_unified/gateway/__init__.py`

- [x] **Step 1: Add bootstrap config and helper**

Implement a small helper that:
- accepts a backend `GatewayRuntime`;
- accepts optional `profile_store`, `profiles`, `default_profile_id`, and `default_preset_id`;
- duplicates `default_preset_id` into a deterministic profile id when no explicit `default_profile_id` is supplied;
- stores caller-supplied profiles before preset defaults so explicit profiles remain addressable;
- returns `ProfileAwareGatewayRuntime` with the selected default profile id.

- [x] **Step 2: Keep exports lazy and dependency-light**

Export the helper from `mcp_unified.gateway` without making stdio imports require FastAPI and without adding host imports.

- [x] **Step 3: Run GREEN tests**

Run the focused gateway package tests and confirm all pass.

Result after implementation and rebase onto `origin/dev` `53d224c4fb`: `41 passed, 4 warnings`.

### Task 3: Compatibility, Security, And PR Handoff

**Files:**
- Modify: `Docs/superpowers/plans/2026-05-30-mcp-unified-stage4f-gateway-profile-bootstrap-plan.md`
- Modify: `backlog/tasks/task-563 - Implement-MCP-Unified-Stage-4F-gateway-profile-bootstrap.md`

- [x] **Step 1: Run host compatibility tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage4f-gateway-profile-bootstrap/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py -q
```

- [x] **Step 2: Run lint, security, and whitespace checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage4f-gateway-profile-bootstrap/.venv/bin/python -m ruff check mcp_unified/gateway tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage4f-gateway-profile-bootstrap/.venv/bin/python -m bandit -r mcp_unified/gateway -f json -o /tmp/bandit_mcp_stage4f_gateway_profile_bootstrap.json
git diff --check
```

- [x] **Step 3: Update plan and Backlog task**

Record RED/GREEN and final verification evidence. Check off acceptance criteria and Definition of Done.

- [x] **Step 4: Commit, push, and open PR**

Commit the plan, gateway/test changes, and Backlog task update together, push the branch, and open a PR against `dev`.

### Task 4: PR Review Collision Fix

**Files:**
- Modify: `mcp_unified/gateway/bootstrap.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
- Modify: `Docs/superpowers/plans/2026-05-30-mcp-unified-stage4f-gateway-profile-bootstrap-plan.md`
- Modify: `backlog/tasks/task-563 - Implement-MCP-Unified-Stage-4F-gateway-profile-bootstrap.md`

- [x] **Step 1: Verify review findings**

Confirmed the Qodo and Gemini comments identify a real policy overwrite risk: preset seeding used the resolved default id, so a caller profile could be replaced by the preset profile.

- [x] **Step 2: Add RED collision coverage**

Added regression tests for an explicit caller default plus seeded preset, and for direct preset-id collision. RED result: `2 failed, 41 passed, 4 warnings`.

- [x] **Step 3: Preserve caller profiles and reject collisions**

Preset seeding now uses the built-in preset id as the seeded profile id and raises `ValueError` if that profile id already exists in the selected store.

- [x] **Step 4: Re-run focused and compatibility validation**

Focused gateway package tests passed with `43 passed, 4 warnings`; compatibility tests passed with `47 passed, 4 warnings`.

## Verification

- Baseline before RED: `38 passed, 4 warnings` for `test_gateway_fastapi_package.py`.
- RED: `3 failed, 38 passed, 4 warnings`; expected missing bootstrap module failures only.
- GREEN after rebase onto `origin/dev` `53d224c4fb`: `41 passed, 4 warnings` for `test_gateway_fastapi_package.py`.
- PR review RED after rebase onto `origin/dev` `7ef87742ac`: `2 failed, 41 passed, 4 warnings`; expected preset overwrite/collision failures only.
- PR review GREEN after collision fix: `43 passed, 4 warnings` for `test_gateway_fastapi_package.py`.
- Compatibility: `47 passed, 4 warnings` for `test_extraction_contracts.py` and `test_http_mapping.py`.
- Ruff: `All checks passed!` for `mcp_unified/gateway` and the focused gateway package tests.
- Bandit: `0` results and no errors for `mcp_unified/gateway`.
- Whitespace: `git diff --check` passed.
