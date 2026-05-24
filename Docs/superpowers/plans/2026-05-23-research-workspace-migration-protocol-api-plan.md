# Research Workspace Migration Protocol API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a durable backend migration protocol for Research Workspace sessions, chunk receipts, recovery manifests, and deletion-ack safety guards.

**Architecture:** Add a small migration router mounted before the dynamic workspace router under `/api/v1/workspaces`. Persist sessions and chunk receipts in ChaChaNotes DB, keep validation in Pydantic/API boundaries plus DB conflict checks, and leave source ingestion/indexing Jobs as a follow-up.

**Tech Stack:** FastAPI, Pydantic v2, SQLite/PostgreSQL-compatible ChaChaNotes DB helpers, pytest `TestClient`, Bandit.

---

### Task 1: Add API Contract Tests

**Files:**
- Create: `tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py`
- Modify: `backlog/completed/task-469 - Implement-Research-Workspace-migration-protocol-API.md`

- [x] **Step 1: Write failing tests for session create/read and route ordering**

```python
def test_workspace_migration_session_create_read_and_route_order(...):
    create = client.post("/api/v1/workspaces/migrations", json={...})
    assert create.status_code == 201
    fetched = client.get("/api/v1/workspaces/migrations/mig-1")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == "mig-1"
```

- [x] **Step 2: Run the test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py::test_workspace_migration_session_create_read_and_route_order -q`

Expected: FAIL because migration endpoints do not exist.

- [x] **Step 3: Add failing tests for chunk idempotency/conflict, finalize, oversize, and delete-ack guard**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py -q`

Expected: FAIL because migration schemas/routes/DB methods do not exist.

### Task 2: Add Schemas and Router Mount

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/workspace_migrations.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`

- [x] **Step 1: Add Pydantic schemas and protocol limits**

Add request/response models for migration sessions, chunks, finalize, and delete ack. Validate SHA-256 format, chunk counts, chunk byte counts, and bounded metadata.

- [x] **Step 2: Add a dedicated `workspace_migrations` router**

Implement route handlers that call DB methods and map `InputError`, `ConflictError`, and `CharactersRAGDBError` through existing HTTP error helpers.

- [x] **Step 3: Mount migration router before workspace router**

Register `tldw_Server_API.app.api.v1.endpoints.workspace_migrations` with prefix `/api/v1/workspaces` before `workspaces` in both minimal and content router groups.

- [x] **Step 4: Run route-ordering test**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py::test_workspace_migration_session_create_read_and_route_order -q`

Expected: FAIL until DB persistence methods exist, but no longer 404s due route ordering.

### Task 3: Add Durable ChaChaNotes Persistence

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py`

- [x] **Step 1: Add v47 migration and schema ensure helpers**

Create `workspace_migration_sessions` and `workspace_migration_chunks` for SQLite and PostgreSQL. Bump `_CURRENT_SCHEMA_VERSION` to 47 and wire v46->v47 migration paths.

- [x] **Step 2: Add DB methods**

Implement:
- `upsert_workspace_migration_session(data)`
- `get_workspace_migration_session(migration_id)`
- `add_workspace_migration_chunk(migration_id, chunk_id, data)`
- `finalize_workspace_migration(migration_id, data)`
- `record_workspace_migration_client_delete_ack(migration_id, data)`

- [x] **Step 3: Enforce DB-level idempotency/conflicts**

Reuse the same migration id with the same idempotency key/manifest hash returns the existing row. Reuse with different key/hash raises `ConflictError`. Duplicate chunks with the same hash/bytes return existing rows. Duplicate chunks with different hash/bytes raise `ConflictError`.

- [x] **Step 4: Run migration API tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py -q`

Expected: PASS.

### Task 4: Verify Broader Workspace Behavior

**Files:**
- Test: `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py`

- [x] **Step 1: Run focused workspace tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_migration_api.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py -q`

Expected: PASS.

- [x] **Step 2: Run Bandit on touched Python files**

Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/workspace_migrations.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/api/v1/schemas/workspace_schemas.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json -o /tmp/bandit_research_workspace_migration_protocol.json`

Expected: no new findings in touched code.

- [x] **Step 3: Validate with a real backend/API run**

Start a local backend with the project venv, call `/api/v1/health`, then call the migration create/get/chunk/finalize endpoints using `curl` against the live server. Confirm `/api/v1/workspaces/migrations` is not treated as a workspace id.

### Task 5: Finalize Tracking

**Files:**
- Modify: `backlog/completed/task-469 - Implement-Research-Workspace-migration-protocol-API.md`

- [x] **Step 1: Update task notes**

Record touched files, verification commands, real backend result, and any known follow-ups.

- [x] **Step 2: Self-review**

Check route ordering, migration idempotency, validation messages, and lack of `/workspace-playground` aliases or redirects.
