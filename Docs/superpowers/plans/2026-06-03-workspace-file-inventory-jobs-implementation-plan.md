# Workspace File Inventory Jobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Jobs-backed, metadata-only file inventory scanner for Workspace primary roots, with durable scan status, bounded diagnostics, redacted item listing, and context/capability integration.

**Architecture:** Add Workspace-owned scan and item tables to `CharactersRAGDB`, enqueue scans through a new Workspace Jobs helper, execute traversal in a focused Jobs worker, and expose status/items through additive `/api/v1/workspaces/{workspace_id}/file-inventory/*` routes. The worker resolves roots from the DB instead of trusting absolute paths in job payloads, and the scanner records metadata only.

**Tech Stack:** FastAPI, Pydantic v2, `CharactersRAGDB`, core Jobs `JobManager`/`WorkerSDK`, SQLite/PostgreSQL-compatible DB abstractions, pytest, Bandit.

---

## Scope Boundaries

In scope:

- Metadata-only file inventory for one primary root per Workspace.
- Durable scan records, item projection, counts, diagnostics, and stale detection.
- Host-local root scan resolution through existing Workspace root-binding policy.
- Fail-closed `sandbox_volume` handling when no mounted-path resolver exists.
- Bounded ignore policy, including built-in generated/secret skips and a conservative `.gitignore` subset.
- Jobs enqueue helper, worker handler, startup worker registration, and API routes.
- Context/capability additions for `scan_files` and `view_file_inventory`.

Out of scope:

- File-content indexing, chunking, embeddings, and RAG.
- UI implementation.
- Git status commands.
- File preview/open/download.
- MCP trusted-root mutation.
- Sandbox lifecycle and volume mount creation.
- Secondary roots.
- Route aliases or redirects.

## File Structure

- Create `tldw_Server_API/app/core/Workspaces/file_inventory_models.py`
  - Dataclasses, state literals, count normalization, diagnostic redaction, cursor helpers.
- Create `tldw_Server_API/app/core/Workspaces/file_inventory_ignore.py`
  - Built-in ignore policy, bounded `.gitignore` subset parser, policy fingerprint.
- Create `tldw_Server_API/app/core/Workspaces/file_inventory_scanner.py`
  - Metadata-only traversal, bounds, symlink skip behavior, item production.
- Create `tldw_Server_API/app/core/Workspaces/file_inventory_jobs.py`
  - Jobs constants, queue env helper, enqueue helper, idempotency behavior.
- Create `tldw_Server_API/app/services/workspace_file_inventory_jobs_worker.py`
  - WorkerSDK entrypoint and job handler.
- Modify `tldw_Server_API/app/core/Workspaces/root_binding_service.py`
  - Expose a reusable host-local root resolution helper or add a small resolver wrapper consumed by the worker.
- Modify `tldw_Server_API/app/core/Workspaces/context.py`
  - Project inventory status into root/capability context.
- Modify `tldw_Server_API/app/core/Workspaces/models.py`
  - Add inventory states and fail-closed action names if needed.
- Modify `tldw_Server_API/app/core/Workspaces/README.md`
  - Document file inventory module boundaries and non-indexing behavior.
- Modify `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Add schema tables and DB methods for scans/items/status.
- Modify `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
  - Add scan request, status response, item response, and nested file inventory fields.
- Modify `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
  - Add scan/status/items endpoints and context integration.
- Modify `tldw_Server_API/app/services/startup_primary_jobs_pollers.py`
  - Register Workspace file inventory Jobs worker behind `WORKSPACE_FILE_INVENTORY_JOBS_WORKER_ENABLED`.
- Test `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_db.py`
- Test `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_ignore.py`
- Test `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_scanner.py`
- Test `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_jobs.py`
- Test `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_worker.py`
- Test `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_api.py`
- Modify `tldw_Server_API/tests/Workspaces/test_workspace_core_context.py`
- Modify `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`
- Modify startup worker tests if present for `startup_primary_jobs_pollers.py`.

## Parallelization Map

Sequential blocker:

- Task 1 must land first because later tasks import shared inventory models.

Can run in parallel after Task 1:

- Task 2 DB scan/item persistence.
- Task 3 ignore policy.
- Task 4 scanner traversal.

Must integrate after Tasks 2-4:

- Task 5 Jobs enqueue helper and worker.
- Task 6 API schemas/endpoints.
- Task 7 context/capabilities/startup integration.

Final:

- Task 8 documentation, security verification, and focused/full test pass.

Avoid parallel edits to `ChaChaNotes_DB.py`, `workspace_schemas.py`, and
`workspaces.py` unless ownership is explicitly split.

---

### Task 1: Inventory Model Contracts

**Files:**
- Create: `tldw_Server_API/app/core/Workspaces/file_inventory_models.py`
- Modify: `tldw_Server_API/app/core/Workspaces/models.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_models.py`

- [ ] **Step 1: Write failing tests for inventory states, counts, diagnostics, and redaction**

Create tests covering:

- valid states: `not_started`, `queued`, `scanning`, `current`, `partial`, `stale`, `failed`, `disabled`
- unknown state normalizes to `failed`
- diagnostic redaction strips absolute paths to relative hints
- diagnostic list is capped at 50
- counts default missing keys to zero

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_models.py -q
```

Expected: FAIL because module does not exist.

- [ ] **Step 2: Implement model helpers**

Add:

- `WorkspaceFileInventoryState`
- `WorkspaceFileInventoryCounts`
- `WorkspaceFileInventoryDiagnostic`
- `normalize_inventory_state(value)`
- `normalize_inventory_counts(value)`
- `bounded_inventory_diagnostics(value, root_relative_only=True)`
- `redact_inventory_path_hint(value)`

Keep helpers free of FastAPI, DB, and filesystem side effects.

- [ ] **Step 3: Run model tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_models.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tldw_Server_API/app/core/Workspaces/file_inventory_models.py tldw_Server_API/app/core/Workspaces/models.py tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_models.py
git commit -m "feat: add workspace file inventory model contracts"
```

---

### Task 2: DB Persistence For Scans And Items

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_db.py`

- [ ] **Step 1: Write failing DB tests**

Cover:

- schema creates `workspace_file_inventory_scans` and `workspace_file_inventory_items`
- `begin_workspace_file_inventory_scan` creates queued scan and updates root `file_inventory_state`
- active queued/scanning scan is reused
- completed scan stores counts and bounded diagnostics
- replacing item projection marks missing previous items as deleted
- status computes `stale` when current root version differs from scan root version
- item list returns relative paths only and respects `include_ignored`
- SQLite uniqueness races map to `ConflictError`/idempotent return

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_db.py -q
```

Expected: FAIL because DB methods/tables do not exist.

- [ ] **Step 2: Add schema tables**

In the existing workspace schema initialization and migration section, add:

- `workspace_file_inventory_scans`
- `workspace_file_inventory_items`
- indexes from the design spec

Keep SQL inside `CharactersRAGDB`.

- [ ] **Step 3: Add DB methods**

Implement:

- `begin_workspace_file_inventory_scan(...)`
- `attach_workspace_file_inventory_job(...)`
- `mark_workspace_file_inventory_scanning(...)`
- `complete_workspace_file_inventory_scan(...)`
- `replace_workspace_file_inventory_items(...)`
- `get_workspace_file_inventory_status(...)`
- `list_workspace_file_inventory_items(...)`

Use existing transaction helpers and catch both `sqlite3` errors and
`BackendDatabaseError` consistently with project-root methods.

- [ ] **Step 4: Run DB tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_db.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_db.py
git commit -m "feat: persist workspace file inventory scans"
```

---

### Task 3: Ignore Policy

**Files:**
- Create: `tldw_Server_API/app/core/Workspaces/file_inventory_ignore.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_ignore.py`

- [ ] **Step 1: Write failing ignore-policy tests**

Cover:

- built-in generated directories are skipped
- secret-like files are skipped
- simple `.gitignore` patterns skip files
- malformed or oversized ignore files produce diagnostics but do not crash
- fingerprint changes when rules change
- fingerprint is stable for equivalent rule ordering

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_ignore.py -q
```

Expected: FAIL because module does not exist.

- [ ] **Step 2: Implement conservative ignore policy**

Use `fnmatch` and `PurePosixPath` for a documented subset:

- blank/comment lines ignored
- trailing slash means directory pattern
- anchored `/path` patterns apply from root
- unanchored patterns match path segments
- `*` and `?` use `fnmatch`

When unsure, prefer skipping with a diagnostic over including a risky path.

- [ ] **Step 3: Run ignore-policy tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_ignore.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tldw_Server_API/app/core/Workspaces/file_inventory_ignore.py tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_ignore.py
git commit -m "feat: add workspace file inventory ignore policy"
```

---

### Task 4: Metadata Scanner

**Files:**
- Create: `tldw_Server_API/app/core/Workspaces/file_inventory_scanner.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_scanner.py`

- [ ] **Step 1: Write failing scanner tests**

Use `tmp_path` fixtures to cover:

- records files/directories with relative POSIX paths
- does not read ordinary file contents
- skips symlinked directories
- records symlink entries without following targets, if supported
- honors built-in ignore policy
- records partial diagnostics for permission/stat failures
- stops at max files/depth/diagnostics bounds
- never emits absolute paths in items or diagnostics

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_scanner.py -q
```

Expected: FAIL because scanner does not exist.

- [ ] **Step 2: Implement scanner**

Add a pure scanner function such as:

```python
def scan_workspace_file_inventory(root: Path, *, policy: InventoryIgnorePolicy, bounds: InventoryScanBounds) -> InventoryScanResult:
    ...
```

Use `os.scandir`/`Path` metadata only. Do not open ordinary files.

- [ ] **Step 3: Run scanner tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_scanner.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tldw_Server_API/app/core/Workspaces/file_inventory_scanner.py tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_scanner.py
git commit -m "feat: scan workspace file metadata"
```

---

### Task 5: Jobs Enqueue Helper And Worker

**Files:**
- Create: `tldw_Server_API/app/core/Workspaces/file_inventory_jobs.py`
- Create: `tldw_Server_API/app/services/workspace_file_inventory_jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Workspaces/root_binding_service.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_jobs.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_worker.py`

- [ ] **Step 1: Write failing Jobs helper tests**

Cover:

- enqueue payload contains no absolute path
- scan DB row is created before job enqueue
- idempotency key uses `scan_id`
- Jobs unavailable maps to explicit failure/exception, not silent success
- active queued scan is reused

- [ ] **Step 2: Write failing worker tests**

Cover:

- unsupported job type fails non-retryably
- malformed payload fails non-retryably
- root version mismatch marks scan stale/failed
- host-local scan completes and writes items
- sandbox root without mounted resolver fails closed
- worker result includes counts and diagnostics

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_jobs.py tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_worker.py -q
```

Expected: FAIL because helper/worker do not exist.

- [ ] **Step 3: Implement enqueue helper**

Add:

- `WORKSPACE_JOBS_DOMAIN = "workspaces"`
- `WORKSPACE_FILE_INVENTORY_JOB_TYPE = "workspace_file_inventory_scan"`
- `workspace_file_inventory_jobs_queue()`
- `enqueue_workspace_file_inventory_scan_job(...)`

Ensure helper returns the scan status and job row.

- [ ] **Step 4: Implement worker**

Use `WorkerSDK` like CodeGraph and media workers. Run filesystem scanning in
`asyncio.to_thread`. Update durable DB scan state before and after traversal.

- [ ] **Step 5: Run Jobs tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_jobs.py tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_worker.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Workspaces/file_inventory_jobs.py tldw_Server_API/app/services/workspace_file_inventory_jobs_worker.py tldw_Server_API/app/core/Workspaces/root_binding_service.py tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_jobs.py tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_worker.py
git commit -m "feat: add workspace file inventory jobs worker"
```

---

### Task 6: API Schemas And Endpoints

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_api.py`

- [ ] **Step 1: Write failing API tests**

Cover:

- `POST /api/v1/workspaces/{workspace_id}/file-inventory/scan`
- `GET /api/v1/workspaces/{workspace_id}/file-inventory/status`
- `GET /api/v1/workspaces/{workspace_id}/file-inventory/items`
- no root returns 409
- root version mismatch returns 409
- Jobs unavailable returns 503
- item responses contain relative paths only
- failed scan diagnostics are bounded and redacted

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_api.py -q
```

Expected: FAIL because routes/schemas do not exist.

- [ ] **Step 2: Add Pydantic schemas**

Add:

- `WorkspaceFileInventoryScanRequest`
- `WorkspaceFileInventoryJobStatus`
- `WorkspaceFileInventoryCounts`
- `WorkspaceFileInventoryDiagnostic`
- `WorkspaceFileInventoryStatusResponse`
- `WorkspaceFileInventoryItemResponse`
- `WorkspaceFileInventoryItemsResponse`

Reuse existing `WorkspaceFileInventory` as the nested summary compatibility
shape.

- [ ] **Step 3: Add endpoints**

Implement the three routes in `workspaces.py`. Keep endpoint logic thin:

- require workspace
- call DB/helper methods
- map Workspace/DB/Jobs errors
- return schemas

- [ ] **Step 4: Run API tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_api.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/workspace_schemas.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_api.py
git commit -m "feat: expose workspace file inventory API"
```

---

### Task 7: Context, Capabilities, Startup, And Docs

**Files:**
- Modify: `tldw_Server_API/app/core/Workspaces/context.py`
- Modify: `tldw_Server_API/app/core/Workspaces/README.md`
- Modify: `tldw_Server_API/app/services/startup_primary_jobs_pollers.py`
- Modify: `tldw_Server_API/tests/Workspaces/test_workspace_core_context.py`
- Modify: `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`
- Modify or create startup worker test for `startup_primary_jobs_pollers.py`

- [ ] **Step 1: Write failing integration tests**

Cover:

- context includes nested file inventory summary
- `scan_files` allowed only for attached roots that are not known unready from
  persisted Workspace state
- `view_file_inventory` allowed for failed scans so diagnostics can be seen
- `index_file_content` remains disabled
- startup registers worker when `WORKSPACE_FILE_INVENTORY_JOBS_WORKER_ENABLED=true`
- startup skips worker when disabled

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_core_context.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q
```

Expected: FAIL for missing context/capability fields.

- [ ] **Step 2: Project inventory into context**

Extend the root projection to include:

- `file_inventory_state`
- nested `file_inventory`
- fail-closed action states for `scan_files`, `view_file_inventory`, and
  unchanged `index_file_content`

- [ ] **Step 3: Register startup worker**

Add the worker to `provide_primary_jobs_worker_specs` and legacy startup helper
paths behind `WORKSPACE_FILE_INVENTORY_JOBS_WORKER_ENABLED`.

- [ ] **Step 4: Update README**

Document:

- metadata-only scan boundary
- no automatic file indexing
- public path redaction
- job worker env flags
- follow-up explicit indexing slice

- [ ] **Step 5: Run integration tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_core_context.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Workspaces/context.py tldw_Server_API/app/core/Workspaces/README.md tldw_Server_API/app/services/startup_primary_jobs_pollers.py tldw_Server_API/tests/Workspaces/test_workspace_core_context.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py
git commit -m "feat: surface workspace file inventory status"
```

---

### Task 8: Verification, Hardening, And Finalization

**Files:**
- Modify as needed from review findings.
- Modify: `backlog/tasks/<task-id>.md` or completed task record through Backlog tooling.

- [ ] **Step 1: Run focused Workspace tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces -q
```

Expected: PASS.

- [ ] **Step 2: Run import/compile check**

```bash
source .venv/bin/activate && python -m compileall -q tldw_Server_API/app/core/Workspaces tldw_Server_API/app/services tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/api/v1/schemas/workspace_schemas.py
```

Expected: exit 0.

- [ ] **Step 3: Run Bandit on touched backend scope**

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Workspaces tldw_Server_API/app/services/workspace_file_inventory_jobs_worker.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/api/v1/schemas/workspace_schemas.py -f json -o /tmp/bandit_workspace_file_inventory.json
```

Expected: no new findings in touched code.

- [ ] **Step 4: Run diff hygiene**

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 5: Update Backlog task with verification**

Record:

- tests run
- Bandit result path
- known skips
- final summary

- [ ] **Step 6: Commit final hardening**

```bash
git status --short
git add <changed-files>
git commit -m "chore: finalize workspace file inventory worker"
```

---

## Review Checklist

- Job payload contains no absolute root path.
- Public responses contain no absolute root path.
- Scanner does not read ordinary file contents.
- Symlink traversal is disabled.
- `.gitignore` support is documented as conservative.
- Bounds produce `partial`, not unhandled exceptions.
- Root replacement or policy change makes old scans stale.
- `index_file_content` remains disabled until explicit indexing exists.
- `sandbox_volume` roots fail closed without a mounted-path resolver.
- SQLite and PostgreSQL/backend abstraction errors map through existing DB error types.
