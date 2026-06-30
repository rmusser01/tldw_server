# MCP Unified Stage 2D Storage Contract Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add standalone package storage contracts for MCP profile assignments, approval policies, credential grants, external registry entries, and audit events without adding persistence or runtime execution wiring.

**Architecture:** This slice keeps all behavior inside the `mcp_unified` package and existing package-boundary tests. It adds typed Pydantic storage payload models and expands storage protocols so future SQLite and host adapters can implement separate stores instead of overloading `ProfileStore`. No FastAPI routes, `MCPProtocol` enforcement, `MCPServer` behavior, SQLite migrations, external process lifecycle, or gateway entrypoints change.

**Tech Stack:** Python 3.10+, Pydantic v2, pytest, Ruff, Mypy, Bandit, Backlog.md.

---

## Source Design

- Spec: `Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md`
- Prior slice: `Docs/superpowers/plans/2026-05-27-mcp-unified-stage2c-structured-resolution-implementation-plan.md`
- Backlog task: `TASK-526`

## Scope

In scope:
- Add package-local storage payload models for assignments, approval policies, credential grants, external server definitions, and audit events.
- Expand `mcp_unified.interfaces.storage` with separate protocols for those stores.
- Export the contracts from `mcp_unified.interfaces`.
- Add tests proving import isolation, safe defaults, timestamp awareness, profile assignment defaults, credential grant non-secret shape, and protocol exports.
- Update Backlog task state and verification evidence.

Out of scope:
- SQLite persistence and migrations.
- Runtime profile enforcement in `MCPProtocol` or `MCPServer`.
- FastAPI route changes.
- MCP Hub/AuthNZ adapter rewiring.
- External MCP process spawning, stdio lifecycle, or gateway entrypoints.

## Files

- Create: `mcp_unified/storage/__init__.py`
- Create: `mcp_unified/storage/models.py`
- Modify: `mcp_unified/interfaces/storage.py`
- Modify: `mcp_unified/interfaces/__init__.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_storage_contracts.py`
- Modify: `backlog/tasks/task-526 - Implement-MCP-Unified-Stage-2D-storage-contract-split.md`

## Task 1: RED Tests For Split Storage Contracts

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_storage_contracts.py`

- [x] **Step 1: Write failing storage-contract tests**

Add tests that assert:
- `mcp_unified.storage` imports without any `tldw_Server_API` imports.
- `ProfileAssignment` preserves principal/workspace/default-profile binding fields and starts enabled.
- `CredentialGrant` stores broker/slot metadata but has no `secret`, `token`, `api_key`, or `value` fields.
- `ExternalServerDefinition` defaults lifecycle metadata safely without spawning anything.
- `AuditEvent` timestamps are timezone-aware and payloads are caller-owned copies.
- `mcp_unified.interfaces.storage` exports `ProfileAssignmentStore`, `ApprovalPolicyStore`, `CredentialGrantStore`, `ExternalRegistryStore`, and `AuditStore`.

- [x] **Step 2: Run RED test**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage2d-runtime-dependencies/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_storage_contracts.py -v
```

Expected: FAIL because `mcp_unified.storage` and the new store protocols do not exist yet.

## Task 2: Add Storage Payload Models

**Files:**
- Create: `mcp_unified/storage/__init__.py`
- Create: `mcp_unified/storage/models.py`

- [x] **Step 1: Implement models**

Add Pydantic models:
- `ProfileAssignment`
- `ApprovalPolicyDocument`
- `CredentialGrant`
- `ExternalServerDefinition`
- `AuditEvent`

Use aware UTC defaults, safe list/dict defaults, and explicit field names that avoid secret material in credential grants.

- [x] **Step 2: Export models**

Export all five models from `mcp_unified.storage`.

- [x] **Step 3: Run storage model tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage2d-runtime-dependencies/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_storage_contracts.py -v
```

Expected: tests still fail until protocol exports are added.

## Task 3: Split Storage Protocols

**Files:**
- Modify: `mcp_unified/interfaces/storage.py`
- Modify: `mcp_unified/interfaces/__init__.py`

- [x] **Step 1: Add protocols**

Keep `ProfileStore` unchanged and add:
- `ProfileAssignmentStore`
- `ApprovalPolicyStore`
- `CredentialGrantStore`
- `ExternalRegistryStore`
- `AuditStore`

Update the existing sparse external/audit protocols to use the typed models.

- [x] **Step 2: Export protocols**

Update `mcp_unified.interfaces.__init__` to export the new store protocols.

- [x] **Step 3: Run GREEN tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage2d-runtime-dependencies/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_storage_contracts.py -v
```

Expected: PASS.

## Task 4: Regression And Quality Gates

**Files:**
- Modify: `backlog/tasks/task-526 - Implement-MCP-Unified-Stage-2D-storage-contract-split.md`

- [x] **Step 1: Run focused regression tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage2d-runtime-dependencies/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_storage_contracts.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py \
  -v
```

Expected: PASS.

- [x] **Step 2: Run static and security checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage2d-runtime-dependencies/.venv/bin/python -m ruff check mcp_unified tldw_Server_API/app/core/MCP_unified/tests/test_storage_contracts.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage2d-runtime-dependencies/.venv/bin/python -m mypy mcp_unified tldw_Server_API/app/core/MCP_unified/tests/test_storage_contracts.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage2d-runtime-dependencies/.venv/bin/python -m bandit -r mcp_unified -f json -o /tmp/bandit_mcp_unified_stage2d_storage.json
jq '.metrics._totals, (.results | length)' /tmp/bandit_mcp_unified_stage2d_storage.json
git diff --check
```

Expected: Ruff passes, Mypy passes, Bandit reports 0 findings, and diff whitespace is clean.

- [x] **Step 3: Update task and commit**

Record implementation notes, verification, known skips, and final summary in `TASK-526`, then commit:

```bash
git add \
  Docs/superpowers/plans/2026-05-28-mcp-unified-stage2d-storage-contract-split-implementation-plan.md \
  mcp_unified/storage/__init__.py \
  mcp_unified/storage/models.py \
  mcp_unified/interfaces/storage.py \
  mcp_unified/interfaces/__init__.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_storage_contracts.py \
  "backlog/tasks/task-526 - Implement-MCP-Unified-Stage-2D-storage-contract-split.md"
git commit -m "feat: add mcp storage contract split"
```
