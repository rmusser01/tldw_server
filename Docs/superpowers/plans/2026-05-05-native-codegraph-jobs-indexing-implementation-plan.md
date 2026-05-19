# Native CodeGraph Jobs Indexing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the deferred Jobs-backed execution path for native CodeGraph index and sync work.

**Architecture:** Keep foreground CodeGraph behavior unchanged. Add small job payload helpers under `app/core/CodeGraph/`, a worker entrypoint that delegates to the existing `CodeGraphIndexer`, and thin MCP enqueue handling in `CodeGraphModule`. The first Jobs slice should not add file watching, Scheduler integration, automatic in-process startup, or new graph semantics.

**Tech Stack:** Python 3.11, Unified MCP `BaseModule`, core Jobs `JobManager`/`WorkerSDK`, SQLite-backed CodeGraph repository, pytest/pytest-asyncio, Ruff, Bandit.

---

## Scope

Implement only:

- `mode="job"` and `mode="background"` for `codegraph.index` and `codegraph.sync`.
- Core Jobs payload helpers for CodeGraph index and sync work.
- A CodeGraph Jobs worker handler and `run_codegraph_jobs_worker()` entrypoint.
- Focused tests for helper payloads, worker validation/execution, MCP enqueue responses, and existing foreground behavior.

Do not implement:

- File watchers or Scheduler-triggered recurring sync.
- Automatic worker startup in FastAPI lifespan.
- Jobs-specific progress callbacks beyond returning the serialized index result.
- Full cancellation checkpoints inside `CodeGraphIndexer`.
- Any new language extractor or semantic-resolution behavior.

## File Structure

- Create `tldw_Server_API/app/core/CodeGraph/jobs.py`
  - Constants for Jobs domain, queue, and job types.
  - JSON-safe payload builder from `WorkspaceResolution`, `CodeGraphSettings`, and tool arguments.
  - `enqueue_codegraph_index_job(...)` helper that accepts an optional `JobManager`.
- Create `tldw_Server_API/app/core/CodeGraph/jobs_worker.py`
  - `handle_codegraph_index_job(job)` worker handler.
  - Safe payload validation and path-bounding for `workspace_root` and `index_db_path`.
  - `run_codegraph_jobs_worker(stop_event=None)` WorkerSDK entrypoint.
- Modify `tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py`
  - Accept optional `job_manager_factory` for tests.
  - Add job/background mode validation.
  - Enqueue Jobs entries for index/sync job mode via `asyncio.to_thread`.
- Add `tldw_Server_API/tests/CodeGraph/test_codegraph_jobs.py`
  - Payload/enqueue helper tests.
- Add `tldw_Server_API/tests/CodeGraph/test_codegraph_jobs_worker.py`
  - Worker handler success and validation tests.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`
  - MCP job-mode enqueue tests and foreground regression.
- Modify `backlog/tasks/task-70 - Add-CodeGraph-Jobs-backed-indexing-mode.md`
  - Record plan, implementation notes, verification, and final summary.

## Task 1: Job Payload Helpers

**Files:**

- Create `tldw_Server_API/app/core/CodeGraph/jobs.py`
- Test `tldw_Server_API/tests/CodeGraph/test_codegraph_jobs.py`

- [x] **Step 1: Write failing payload/enqueue tests**

Add tests proving:

- `build_codegraph_index_job_payload(...)` returns JSON-safe strings for paths, settings, languages, `force`, and operation.
- `enqueue_codegraph_index_job(...)` creates a Jobs row with domain `codegraph`, queue from `CODEGRAPH_JOBS_QUEUE` or `default`, job type `codegraph_index`, owner id, and payload.

Run:

```bash
python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_jobs.py -q
```

Expected: fail because `tldw_Server_API.app.core.CodeGraph.jobs` does not exist.

- [x] **Step 2: Implement minimal helpers**

Create constants:

- `CODEGRAPH_JOBS_DOMAIN = "codegraph"`
- `CODEGRAPH_INDEX_JOB_TYPE = "codegraph_index"`

Implement queue resolution, settings serialization, payload building, and enqueueing through `JobManager.create_job(...)`.

- [x] **Step 3: Verify helper tests pass**

Run:

```bash
python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_jobs.py -q
```

Expected: pass.

## Task 2: Jobs Worker Handler

**Files:**

- Create `tldw_Server_API/app/core/CodeGraph/jobs_worker.py`
- Test `tldw_Server_API/tests/CodeGraph/test_codegraph_jobs_worker.py`

- [x] **Step 1: Write failing worker tests**

Add tests proving:

- A valid index job indexes a Python fixture and returns serialized CodeGraph result data.
- A valid sync job delegates to sync and returns serialized result data.
- Unsupported job types, missing operation, unsupported operation, missing paths, and unsafe index paths fail with non-retryable errors.

Run:

```bash
python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_jobs_worker.py -q
```

Expected: fail because the worker module does not exist.

- [x] **Step 2: Implement worker handler and entrypoint**

Use `CodeGraphSettings.from_mapping(...)`, `CodeGraphLanguageRegistry`, `CodeGraphRepository`, and `CodeGraphIndexer`. Validate that `index_db_path` is below `settings.index_base_dir`; reject unsafe payloads before opening SQLite. Add `run_codegraph_jobs_worker(stop_event=None)` using `WorkerSDK`.

- [x] **Step 3: Verify worker tests pass**

Run:

```bash
python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_jobs_worker.py -q
```

Expected: pass.

## Task 3: MCP Job Mode

**Files:**

- Modify `tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`

- [x] **Step 1: Write failing MCP tests**

Add tests proving:

- `codegraph.index` with `mode="job"` returns `status="queued"`, job identifiers, workspace metadata, and does not initialize the CodeGraph index DB.
- `codegraph.sync` with `mode="background"` enqueues the sync operation.
- Existing `mode="foreground"` behavior still indexes immediately.
- Unknown modes are still rejected.

Run:

```bash
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py -q
```

Expected: fail because only foreground mode is supported.

- [x] **Step 2: Implement MCP enqueue path**

Preserve foreground dispatch. For job/background mode, call `enqueue_codegraph_index_job(...)` through `asyncio.to_thread` with `owner_user_id` derived from the request context. Return a compact MCP response with `status`, `job_id`, `job_uuid`, `job_status`, `workspace_key`, and `mode`.

- [x] **Step 3: Verify MCP tests pass**

Run:

```bash
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py -q
```

Expected: pass.

## Task 4: Final Verification And Task Closeout

**Files:**

- Modify `backlog/tasks/task-70 - Add-CodeGraph-Jobs-backed-indexing-mode.md`

- [x] **Step 1: Run focused tests**

```bash
python -m pytest \
  tldw_Server_API/tests/CodeGraph/test_codegraph_jobs.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_jobs_worker.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py \
  -q
```

- [x] **Step 2: Run Ruff**

```bash
python -m ruff check \
  tldw_Server_API/app/core/CodeGraph/jobs.py \
  tldw_Server_API/app/core/CodeGraph/jobs_worker.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_jobs.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_jobs_worker.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
```

- [x] **Step 3: Run Bandit**

```bash
python -m bandit -r \
  tldw_Server_API/app/core/CodeGraph/jobs.py \
  tldw_Server_API/app/core/CodeGraph/jobs_worker.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py \
  -f json -o /tmp/bandit_codegraph_jobs_indexing.json
```

- [x] **Step 4: Run whitespace check**

```bash
git diff --check
```

- [x] **Step 5: Update TASK-70 and commit**

Record verification, mark acceptance criteria and DoD complete, then commit with:

```bash
git add Docs/superpowers/plans/2026-05-05-native-codegraph-jobs-indexing-implementation-plan.md \
  'backlog/tasks/task-70 - Add-CodeGraph-Jobs-backed-indexing-mode.md' \
  tldw_Server_API/app/core/CodeGraph/jobs.py \
  tldw_Server_API/app/core/CodeGraph/jobs_worker.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_jobs.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_jobs_worker.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
git commit -m "feat: add codegraph jobs indexing mode"
```
