# Research Workspace Source Ingest Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `workspace_source_ingest` Jobs created by Research Workspace source-add safe for the media ingest worker to process.

**Architecture:** Keep source-add job creation unchanged, but route `workspace_source_ingest` through a dedicated readiness-bridge handler inside `media_ingest_jobs_worker.py`. The handler validates a bounded payload, inspects existing Media DB readiness, reports progress using standardized messages, and completes with a small diagnostic result; it does not re-ingest content or write ChaCha workspace state.

**Tech Stack:** FastAPI workspace endpoints, Jobs `JobManager`/`WorkerSDK`, Media DB API, pytest, Bandit.

---

### Task 1: Add Worker Regression Tests

**Files:**
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py`
- Modify: `tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py`

- [ ] **Step 1: Write failing tests**

Add tests that prove:
- `_handle_job()` accepts `workspace_source_ingest` and does not call the ingestion processors.
- A ready media row completes with bounded readiness result and standardized progress.
- A missing media row raises `MediaIngestJobError` with `retryable=False`.
- Completed workspace jobs do not override Media DB readiness in the source status projection.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py \
  tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py \
  -q
```

Expected: Failures showing `workspace_source_ingest` is unsupported and completed-job projection behavior is not yet explicit.

### Task 2: Implement Workspace Source Readiness Bridge

**Files:**
- Modify: `tldw_Server_API/app/services/media_ingest_jobs_worker.py`

- [ ] **Step 1: Add constants and helpers**

Add `_WORKSPACE_SOURCE_JOB_TYPE = "workspace_source_ingest"` and helper functions to:
- parse required workspace payload fields;
- inspect media readiness from Media DB;
- produce a small result payload without raw DB content or errors.

- [ ] **Step 2: Route job types**

Update `_handle_job()` to dispatch `workspace_source_ingest` to `_handle_workspace_source_job()` before enforcing `media_ingest_item`.

- [ ] **Step 3: Preserve existing media ingest behavior**

Run existing worker tests and ensure old `media_ingest_item` behavior remains unchanged.

### Task 3: Verify Status Projection Contract

**Files:**
- Modify only if tests require it: `tldw_Server_API/app/core/Workspaces/status_projection.py`

- [ ] **Step 1: Confirm completed jobs fall through to Media DB**

The projection should use active/failed/retrying job state, but completed jobs should not become the authoritative readiness source.

- [ ] **Step 2: Adjust only if needed**

If current behavior already satisfies this, leave production code unchanged and keep the regression test.

### Task 4: Backend Verification

**Files:**
- Modify: `backlog/tasks/task-500 - Handle-Research-Workspace-source-ingest-jobs-in-media-worker.md`

- [ ] **Step 1: Run focused tests**

Run the worker/status tests.

- [ ] **Step 2: Run broader Workspaces tests**

Run Workspaces API/status tests to catch regressions.

- [ ] **Step 3: Run Bandit**

Run Bandit on touched Python production files.

- [ ] **Step 4: Run live backend smoke**

Start FastAPI and smoke-test source add/status to confirm API behavior remains valid with a real backend.

- [ ] **Step 5: Update Backlog task**

Record implementation notes, verification output, and final summary.
