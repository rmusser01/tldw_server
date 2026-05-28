# MCP Unified Stage 2E SQLite Stores Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add package-local SQLite store primitives for MCP profile documents, profile assignments, approval policies, credential grants, external server definitions, and audit events.

**Architecture:** This slice adds an optional standalone persistence backend under `mcp_unified.storage` that implements the split store protocols from Stage 2D with a schema-versioned SQLite database and JSON payload columns. It must stay package-local: no `tldw_Server_API` imports, no FastAPI routes, no runtime execution enforcement, no external process lifecycle, and no standalone gateway entrypoint.

**Tech Stack:** Python 3.10+, stdlib `sqlite3`, Pydantic v2 models, pytest, Ruff, Mypy, Bandit, Backlog.md.

---

## Source Design

- Spec: `Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md`
- Prior slice: `Docs/superpowers/plans/2026-05-28-mcp-unified-stage2d-storage-contract-split-implementation-plan.md`
- Backlog task: `TASK-528`

## Scope

In scope:
- Add `SQLiteMCPStore` as a package-local implementation of the existing split store protocols.
- Store complete Pydantic payloads as JSON while maintaining indexed columns needed for current protocol filters.
- Add schema metadata with an explicit schema version and idempotent initialization.
- Preserve caller-owned returned models with deep validation/copy boundaries.
- Add focused tests for schema creation, CRUD/filter behavior, audit append/query behavior, and package-boundary isolation.

Out of scope:
- Runtime profile enforcement in `MCPProtocol` or `MCPServer`.
- FastAPI routes, MCP Hub adapters, AuthNZ wiring, or gateway entrypoints.
- External MCP process spawning or lifecycle management.
- YAML import/export or migration CLI commands.
- Cross-process locking beyond SQLite's normal transaction behavior.

## Files

- Create: `mcp_unified/storage/sqlite.py`
- Modify: `mcp_unified/storage/__init__.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py`
- Modify: `Docs/superpowers/plans/2026-05-28-mcp-unified-stage2e-sqlite-stores-implementation-plan.md`
- Modify: `backlog/tasks/task-528 - Implement-MCP-Unified-Stage-2E-SQLite-store-slice.md`

## Task 1: RED Tests For SQLite Store Contracts

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py`

- [x] **Step 1: Write failing import and schema tests**

Add tests that assert:
- `mcp_unified.storage.sqlite` imports without any `tldw_Server_API` imports.
- `SQLiteMCPStore(tmp_path / "mcp.sqlite")` creates schema metadata with `schema_version == "1"`.
- Reopening the same database is idempotent and does not duplicate or corrupt metadata.

- [x] **Step 2: Run RED test**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py -v
```

Expected: FAIL because `mcp_unified.storage.sqlite` and `SQLiteMCPStore` do not exist yet.

## Task 2: Implement Schema Initialization

**Files:**
- Create: `mcp_unified/storage/sqlite.py`
- Modify: `mcp_unified/storage/__init__.py`

- [x] **Step 1: Add SQLiteMCPStore constructor and schema**

Implement:
- `SQLiteMCPStore(path: str | Path)`
- `SCHEMA_VERSION = 1`
- idempotent schema creation for:
  - `mcp_storage_meta`
  - `mcp_profiles`
  - `mcp_profile_assignments`
  - `mcp_approval_policies`
  - `mcp_credential_grants`
  - `mcp_external_servers`
  - `mcp_audit_events`
- `close()` and `async aclose()` helpers.

Use parameterized SQL only. Keep the module free of host imports.

- [x] **Step 2: Export the store**

Export `SQLiteMCPStore` from `mcp_unified.storage`.

- [x] **Step 3: Run schema tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py -v
```

Expected: schema/import tests pass; later CRUD tests will still fail until implemented.

## Task 3: Implement Profile And Split Store CRUD

**Files:**
- Modify: `mcp_unified/storage/sqlite.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py`

- [x] **Step 1: Write failing CRUD/filter tests**

Add tests that assert:
- `upsert_profile`, `get_profile`, `list_profiles`, and `delete_profile` round-trip `MCPProfile` with copy isolation.
- `upsert_assignment`, `get_assignment`, `list_assignments`, and `delete_assignment` filter by `profile_id`, `principal_id`, and `workspace_id`.
- `upsert_policy`, `get_policy`, `list_policies`, and `delete_policy` filter by `profile_id`.
- `upsert_grant`, `get_grant`, `list_grants`, and `delete_grant` filter by `profile_id` and `external_server_id`.
- `upsert_server`, `get_server`, `list_servers`, `list_server_definitions`, and `delete_server` filter enabled definitions separately from status-row compatibility.

- [x] **Step 2: Run RED CRUD/filter tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py -v
```

Expected: FAIL on missing CRUD/filter methods.

- [x] **Step 3: Implement CRUD/filter methods**

Implement protocol methods using model validation at both input and output boundaries. Store full model payloads as sorted JSON and keep filter columns in sync with the payload.

- [x] **Step 4: Run GREEN CRUD/filter tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py -v
```

Expected: PASS for schema and CRUD/filter tests.

## Task 4: Implement Audit Append And Query

**Files:**
- Modify: `mcp_unified/storage/sqlite.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py`

- [x] **Step 1: Write failing audit tests**

Add tests that assert:
- `append_event` persists `AuditEvent` payloads without sharing caller-owned dictionaries.
- `query_events(actor_id=...)`, `query_events(profile_id=...)`, and `query_events(event_type=...)` filter correctly.
- `query_events(limit=...)` returns deterministic newest-first results.

- [x] **Step 2: Run RED audit tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py -v
```

Expected: FAIL on missing audit behavior.

- [x] **Step 3: Implement audit append/query**

Implement append-only insert and parameterized filtered queries. Keep the method async to satisfy the package protocol, but use stdlib SQLite internally for this first package-local backend.

- [x] **Step 4: Run GREEN audit tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py -v
```

Expected: PASS.

## Task 5: Focused Regression And Quality Gates

**Files:**
- Modify: `backlog/tasks/task-528 - Implement-MCP-Unified-Stage-2E-SQLite-store-slice.md`
- Modify: `Docs/superpowers/plans/2026-05-28-mcp-unified-stage2e-sqlite-stores-implementation-plan.md`

- [x] **Step 1: Run focused package regression**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_storage_contracts.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_structured_resolution.py \
  -v
```

Expected: PASS.

- [x] **Step 2: Run static and security checks**

Run:

```bash
source .venv/bin/activate && python -m ruff check mcp_unified tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py
source .venv/bin/activate && python -m mypy mcp_unified tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py
source .venv/bin/activate && python -m bandit -r mcp_unified -f json -o /tmp/bandit_mcp_unified_stage2e_sqlite.json
source .venv/bin/activate && python -c 'import json; data=json.load(open("/tmp/bandit_mcp_unified_stage2e_sqlite.json")); print(data["metrics"]["_totals"]); print("results", len(data["results"]))'
git diff --check
```

Expected: Ruff passes, Mypy passes, Bandit reports 0 findings, and diff whitespace is clean.

- [x] **Step 3: Update task and commit**

Record implementation notes, verification, known skips, and final summary in `TASK-528`, then commit:

```bash
git add \
  Docs/superpowers/plans/2026-05-28-mcp-unified-stage2e-sqlite-stores-implementation-plan.md \
  mcp_unified/storage/__init__.py \
  mcp_unified/storage/sqlite.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py \
  "backlog/tasks/task-528 - Implement-MCP-Unified-Stage-2E-SQLite-store-slice.md"
git commit -m "feat: add mcp sqlite storage contracts"
```
