# Research Core Review Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify and fix validated Research core review findings from TASK-9922.
**Status:** Complete.

**Architecture:** Keep fixes inside the existing Research service, worker, artifact store, and phase handlers. Use owner-scoped service helpers, per-user worker path resolution, immutable artifact file paths, cooperative phase cancellation, and budget checks at provider-call boundaries.

**Tech Stack:** Python, FastAPI service layer, SQLite-backed `ResearchSessionsDB`, pytest, Bandit.

---

### Task 1: Owner Scoping and Checkpoint Replay Guards

**Status:** Complete.

**Files:**
- Modify: `tldw_Server_API/app/core/Research/service.py`
- Test: `tldw_Server_API/tests/Research/test_research_core_hardening.py`

- [x] **Step 1: Write failing tests**

Add tests showing that a shared `ResearchService(research_db_path=...)` rejects cross-owner `get_session`, `get_artifact`, `pause_run`, `resume_run`, `cancel_run`, `build_package`, and `approve_checkpoint`, and that stale or already-approved checkpoints cannot be replayed.

- [x] **Step 2: Verify red**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_core_hardening.py -q
```

Expected: the new owner-scope and stale-checkpoint tests fail against the current code.

- [x] **Step 3: Implement minimal fix**

Add an internal `_get_owned_session(db, owner_user_id, session_id)` helper in `ResearchService` and use it from public methods. Add `_validate_checkpoint_approval_state(...)` to require owner match, pending checkpoint, latest checkpoint, `waiting_human` status, and the expected phase for each checkpoint type before patching or enqueueing.

- [x] **Step 4: Verify green**

Run the same pytest command and confirm the new tests pass.

### Task 2: Worker Per-User Store Resolution

**Status:** Complete.

**Files:**
- Modify: `tldw_Server_API/app/core/Research/jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Research/jobs.py`
- Test: `tldw_Server_API/tests/Research/test_research_core_hardening.py`

- [x] **Step 1: Write failing tests**

Add tests showing `run_research_jobs_worker` handler path resolution uses `job["owner_user_id"]` to derive `DatabasePaths.get_research_sessions_db_path(owner)` and `DatabasePaths.get_user_outputs_dir(owner)` when explicit worker override paths are not supplied.

- [x] **Step 2: Verify red**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_core_hardening.py -q
```

Expected: the worker resolution test fails against the current fixed shared-path behavior.

- [x] **Step 3: Implement minimal fix**

Allow `run_research_jobs_worker` to pass explicit paths only when supplied by arguments or environment. Otherwise resolve per-job paths from `owner_user_id`; if owner is absent, raise a deterministic `ValueError`.

- [x] **Step 4: Verify green**

Run the same pytest command and confirm the new test passes.

### Task 3: Immutable Artifact Versions

**Status:** Complete.

**Files:**
- Modify: `tldw_Server_API/app/core/Research/artifact_store.py`
- Modify if needed: `tldw_Server_API/app/core/DB_Management/ResearchSessionsDB.py`
- Test: `tldw_Server_API/tests/Research/test_research_core_hardening.py`

- [x] **Step 1: Write failing tests**

Add a test that writes two versions of the same artifact and asserts each DB row points to a different existing file whose content matches the row checksum and original payload.

- [x] **Step 2: Verify red**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_core_hardening.py -q
```

Expected: the test fails because both rows currently point at one overwritten path.

- [x] **Step 3: Implement minimal fix**

Compute `next_version` before writing and write to a filename containing `.v<version>` before the extension, using a temporary file and `replace()` for atomic publication.

- [x] **Step 4: Verify green**

Run the same pytest command and confirm the new test passes.

### Task 4: Cooperative Cancellation and Budget Limits

**Status:** Complete.

**Files:**
- Modify: `tldw_Server_API/app/core/Research/jobs.py`
- Modify: `tldw_Server_API/app/core/Research/limits.py`
- Test: `tldw_Server_API/tests/Research/test_research_core_hardening.py`

- [x] **Step 1: Write failing tests**

Add tests showing collection stops before the second focus-area provider call when the first call marks `cancel_requested`, and collection raises a structured limit error when configured `max_searches` is exhausted.

- [x] **Step 2: Verify red**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_core_hardening.py -q
```

Expected: cancellation and limit tests fail against the current phase handler.

- [x] **Step 3: Implement minimal fix**

Add `_halt_for_control_between_steps(...)`, load `ResearchLimits` from `session.limits_json`, check runtime and search budgets before provider calls, and return/cancel promptly between focus areas.

- [x] **Step 4: Verify green**

Run the same pytest command and confirm the new tests pass.

### Task 5: Final Verification and Task Update

**Status:** Complete.

**Files:**
- Modify: `backlog/tasks/task-9922 - Harden-Research-core-review-findings.md`

- [x] **Step 1: Run focused tests**

```bash
./.venv/bin/python -m pytest -p no:unraisableexception tldw_Server_API/tests/Research/test_research_core_hardening.py -q
```

- [x] **Step 2: Run focused compatibility tests**

```bash
./.venv/bin/python -m pytest -p no:unraisableexception tldw_Server_API/tests/Research/test_research_artifact_store.py::test_write_json_artifact_records_manifest tldw_Server_API/tests/Research/test_research_jobs_service.py::test_approve_plan_review_enqueues_collecting_job tldw_Server_API/tests/Research/test_research_jobs_service.py::test_approve_sources_review_enqueues_synthesizing_job_and_writes_review_artifact tldw_Server_API/tests/Research/test_research_jobs_service.py::test_approve_sources_review_with_recollect_enqueues_collecting_job tldw_Server_API/tests/Research/test_research_jobs_service.py::test_approve_outline_review_enqueues_packaging_job tldw_Server_API/tests/Research/test_research_jobs_service.py::test_approve_outline_review_patch_enqueues_resynthesis_with_locked_outline tldw_Server_API/tests/Research/test_research_jobs_service.py::test_get_session_bundle_and_allowlisted_artifacts tldw_Server_API/tests/Research/test_research_jobs_service.py::test_pause_run_marks_active_executable_session_pause_requested tldw_Server_API/tests/Research/test_research_jobs_service.py::test_resume_run_reenqueues_executable_phase_and_restores_checkpoint_wait tldw_Server_API/tests/Research/test_research_jobs_service.py::test_cancel_run_requests_active_work_and_terminalizes_idle_sessions -q
```

- [x] **Step 3: Run Bandit on touched production scope**

```bash
./.venv/bin/python -m bandit -r tldw_Server_API/app/core/Research/service.py tldw_Server_API/app/core/Research/artifact_store.py tldw_Server_API/app/core/Research/jobs.py tldw_Server_API/app/core/Research/jobs_worker.py -f json -o /tmp/bandit_research_core_hardening.json
```

- [x] **Step 4: Run diff hygiene**

```bash
git diff --check -- tldw_Server_API/app/core/Research/service.py tldw_Server_API/app/core/Research/artifact_store.py tldw_Server_API/app/core/Research/jobs.py tldw_Server_API/app/core/Research/jobs_worker.py tldw_Server_API/tests/Research/test_research_core_hardening.py Docs/superpowers/plans/2026-06-23-research-core-review-hardening-plan.md "backlog/tasks/task-9922 - Harden-Research-core-review-findings.md"
```

- [x] **Step 5: Update Backlog task**

Record validated findings, changed files, test results, Bandit result, known skips, and final summary in TASK-9922.
