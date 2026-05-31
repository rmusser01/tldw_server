# Remaining Phase 3.3 Parallel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the remaining safe, covered `Phase 3.3` sanitizer tranches in parallel from the current `phase3.3-error-handler-adoption` worktree state, while preserving public error contracts and avoiding overlapping edits.

**Architecture:** Start by checkpointing the already-verified local service edits into a clean implementation baseline commit. Then run one medium endpoint lane (`sync.py`), two small-service lanes, and one scout lane in parallel. The parent integrates only independently verified shards, records scout defer/reject decisions in the Phase 3.3 plan, and commits by merged wave.

**Tech Stack:** Python 3.11, FastAPI, pytest, loguru, Bandit, SQLite/PostgreSQL-backed service helpers, git worktrees.

---

## File Structure And Ownership

**Checkpoint batch already in local working tree**

- Modify: `tldw_Server_API/app/services/outputs_purge_scheduler.py`
- Modify: `tldw_Server_API/app/services/media_files_cleanup_service.py`
- Modify: `tldw_Server_API/app/services/file_artifacts_export_gc_service.py`
- Modify: `tldw_Server_API/app/services/ingestion_sources_cleanup_service.py`
- Modify: `tldw_Server_API/tests/Services/test_outputs_purge_scheduler_truthiness.py`
- Modify: `tldw_Server_API/tests/Services/test_media_files_cleanup_service.py`
- Create: `tldw_Server_API/tests/Services/test_file_artifacts_export_gc_service.py`
- Modify: `tldw_Server_API/tests/Ingestion_Sources/test_ingestion_sources_cleanup_service.py`
- Modify: `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md`

**Wave 1, Lane A: medium endpoint**

- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Test: `tldw_Server_API/tests/MediaDB2/test_sync_endpoint_errors.py`

**Wave 1, Lane B: small service**

- Modify: `tldw_Server_API/app/services/claims_alerts_scheduler.py`
- Create: `tldw_Server_API/tests/Services/test_claims_alerts_scheduler.py`
- Read-only context: `tldw_Server_API/tests/Services/test_service_startup_truthiness_batch2.py`

**Wave 1, Lane C: small service**

- Modify: `tldw_Server_API/app/services/claims_review_metrics_scheduler.py`
- Create: `tldw_Server_API/tests/Services/test_claims_review_metrics_scheduler.py`
- Read-only context: `tldw_Server_API/tests/Services/test_service_startup_truthiness_batch2.py`

**Wave 1, Lane D: scout**

- Read-only scan targets:
  - `tldw_Server_API/app/services/kanban_activity_cleanup_service.py`
  - `tldw_Server_API/app/services/workflows_webhook_dlq_service.py`
  - `tldw_Server_API/app/services/storage_cleanup_service.py`
  - other `app/services` small files surfaced by the current raw-fallback grep
- Read-only test targets:
  - `tldw_Server_API/tests/Services/test_service_startup_truthiness_batch2.py`
  - module-specific focused tests when they exist
- Modify later by parent only: `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md`

**Parent integration and reporting**

- Modify: `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md`
- Read-only: `Docs/superpowers/specs/2026-04-28-remaining-phase3-3-parallel-design.md`

---

### Task 1: Checkpoint The Current Local Baseline

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md`
- Stage existing touched files listed in the checkpoint batch above

- [ ] **Step 1: Review the current dirty baseline**

Run:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption status --short --branch
```

Expected: only the current local checkpoint files are dirty.

- [ ] **Step 2: Re-run the parent checkpoint verification**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/tests/Services/test_outputs_purge_scheduler_truthiness.py \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/tests/Services/test_media_files_cleanup_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/tests/Services/test_file_artifacts_export_gc_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/tests/Files/test_files_export_gc.py \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/tests/Ingestion_Sources/test_ingestion_sources_cleanup_service.py
```

Expected: the current checkpoint suite passes cleanly.

- [ ] **Step 3: Re-run Bandit on the checkpoint source scope**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m bandit -r \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/app/services/outputs_purge_scheduler.py \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/app/services/media_files_cleanup_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/app/services/file_artifacts_export_gc_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/app/services/ingestion_sources_cleanup_service.py \
  -f json -o /tmp/bandit_phase3_3_checkpoint.json
```

Expected: JSON written successfully with no new findings.

- [ ] **Step 4: Run diff hygiene checks**

Run:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption diff --check
```

Expected: no output.

- [ ] **Step 5: Update the Phase 3.3 plan with the checkpoint summary**

Add a `**Recent Update**` entry covering:

- outputs scheduler remaining failure-path sanitizers
- media cleanup loop result sanitizer
- file export GC sanitizer tranche
- ingestion cleanup sanitizer tranche
- current checkpoint suite passes
- Bandit clean
- `git diff --check` clean

- [ ] **Step 6: Commit the checkpoint**

Run:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption add \
  Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md \
  tldw_Server_API/app/services/outputs_purge_scheduler.py \
  tldw_Server_API/app/services/media_files_cleanup_service.py \
  tldw_Server_API/app/services/file_artifacts_export_gc_service.py \
  tldw_Server_API/app/services/ingestion_sources_cleanup_service.py \
  tldw_Server_API/tests/Services/test_outputs_purge_scheduler_truthiness.py \
  tldw_Server_API/tests/Services/test_media_files_cleanup_service.py \
  tldw_Server_API/tests/Services/test_file_artifacts_export_gc_service.py \
  tldw_Server_API/tests/Ingestion_Sources/test_ingestion_sources_cleanup_service.py
git -C /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption commit -m "Phase 3.3: checkpoint scheduler sanitizer wave"
```

Expected: one clean baseline commit for all further lanes.

---

### Task 2: Create The Wave 1 Candidate Matrix And Scout Artifact

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md`

- [ ] **Step 1: Capture the remaining raw-fallback audit**

Run:

```bash
rg -n "logger\\.(debug|info|warning|error)\\(f\\\".*\\{(e|exc|err|error)\\}|logger\\.(debug|info|warning|error)\\(.*\\{\\}, .*exc|detail=.*\\{(e|exc|err|error)\\}" \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/app/services \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/app/api/v1/endpoints/sync.py
```

Expected: a concrete list of remaining candidate branches.

- [ ] **Step 2: Record the scout input set in the Phase 3.3 plan**

Add a short section or `**Recent Update**` note listing:

- the grep command used
- the Wave 1 approved lanes
- the deferred candidates the scout should inspect next

- [ ] **Step 3: Bless the initial Wave 1 lanes**

Wave 1 should start with:

- Lane A: `sync.py`
- Lane B: `claims_alerts_scheduler.py`
- Lane C: `claims_review_metrics_scheduler.py`
- Lane D scout target set: `kanban_activity_cleanup_service.py`, `workflows_webhook_dlq_service.py`, `storage_cleanup_service.py`

- [ ] **Step 4: Define scout output format**

The scout lane must return, for each candidate:

```text
approve now | defer | reject
source file:
exact branches:
existing tests reviewed:
why conservative or not:
```

- [ ] **Step 5: Do not let the scout edit the shared plan**

The parent copies all scout `defer` and `reject` decisions into the Phase 3.3 plan before reusing Lane D.

---

### Task 3: Lane A Implement `sync.py` Narrowly

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Test: `tldw_Server_API/tests/MediaDB2/test_sync_endpoint_errors.py`

- [ ] **Step 1: Add or extend focused failing tests for log/fallback sanitization only**

Add direct regressions for the covered branches you plan to touch, for example:

```python
@pytest.mark.asyncio
async def test_send_changes_db_error_log_is_sanitized(memory_db_factory, monkeypatch):
    ...
    assert exc_info.value.detail == "Failed to retrieve changes from database."
```

Also add log assertions so raw exception text, user names, and client identifiers are not echoed in fallback logs when that is the intended Phase 3.3 change.

- [ ] **Step 2: Run the focused red selection**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/tests/MediaDB2/test_sync_endpoint_errors.py \
  -k "sanitizes and (send_changes or receive_changes)"
```

Expected: fail specifically because current logs/fallback branches still leak raw details.

- [ ] **Step 3: Patch only the covered fallback/log branches**

Allowed changes:

- replace raw `f"...{e}"` style logging on covered branches
- bind `error_type`
- preserve existing HTTP status codes and response details already pinned by tests

Explicitly do not:

- extract helpers
- refactor query logic
- change response schemas
- widen the touched-file set

- [ ] **Step 4: Run the focused green selection**

Run the same command as Step 2.

Expected: pass.

- [ ] **Step 5: Run the full touched file**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/tests/MediaDB2/test_sync_endpoint_errors.py
```

Expected: full sync error-contract file passes.

- [ ] **Step 6: Run Bandit on `sync.py`**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m bandit -r \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/app/api/v1/endpoints/sync.py \
  -f json -o /tmp/bandit_phase3_3_sync.json
```

- [ ] **Step 7: Stop and hand back**

Return:

- files changed
- red proof
- green proof
- full-file green result
- Bandit result path
- any nearby `sync.py` candidates intentionally skipped

Do not stage or commit in the shared worktree.

---

### Task 4: Lane B Implement `claims_alerts_scheduler.py`

**Files:**
- Modify: `tldw_Server_API/app/services/claims_alerts_scheduler.py`
- Create: `tldw_Server_API/tests/Services/test_claims_alerts_scheduler.py`
- Read-only context: `tldw_Server_API/tests/Services/test_service_startup_truthiness_batch2.py`

- [ ] **Step 1: Write direct failing tests for the remaining raw branches**

Cover these sites:

- `_enumerate_sqlite_user_ids()` base-dir failure
- `_enumerate_sqlite_user_ids()` single-user fallback failure
- `run_claims_alerts_once()` media-db creation failure
- `run_claims_alerts_once()` per-user evaluation failure
- `start_claims_alerts_scheduler()` loop failure

Use a local logger stub like:

```python
class _LoggerStub:
    ...
```

and assert safe messages plus `error_type` bindings.

- [ ] **Step 2: Run the focused red file**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/tests/Services/test_claims_alerts_scheduler.py
```

Expected: fail because current logs interpolate raw exception text.

- [ ] **Step 3: Patch only the covered log branches**

Use the existing Phase 3.3 pattern:

```python
logger.bind(error_type=type(exc).__name__).warning("claims_alerts: failed to create media db")
```

Preserve:

- scheduler enable/disable behavior
- counts/return values
- PostgreSQL vs SQLite control flow

- [ ] **Step 4: Run focused green**

Run the same command as Step 2.

- [ ] **Step 5: Run adjacent startup truthiness coverage**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/tests/Services/test_service_startup_truthiness_batch2.py \
  -k "CLAIMS_ALERTS_SCHEDULER_ENABLED"
```

- [ ] **Step 6: Run Bandit and hand back**

Run Bandit on `claims_alerts_scheduler.py`, then return the same handoff package as Lane A.

---

### Task 5: Lane C Implement `claims_review_metrics_scheduler.py`

**Files:**
- Modify: `tldw_Server_API/app/services/claims_review_metrics_scheduler.py`
- Create: `tldw_Server_API/tests/Services/test_claims_review_metrics_scheduler.py`
- Read-only context: `tldw_Server_API/tests/Services/test_service_startup_truthiness_batch2.py`

- [ ] **Step 1: Write direct failing tests for the remaining raw branches**

Cover these sites:

- `_enumerate_sqlite_user_ids()` base-dir failure
- `_enumerate_sqlite_user_ids()` single-user fallback failure
- `run_claims_review_metrics_once()` media-db creation failure
- `run_claims_review_metrics_once()` per-user aggregation failure
- `start_claims_review_metrics_scheduler()` loop failure

Use the same logger-stub pattern as Task 4.

- [ ] **Step 2: Run the focused red file**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/tests/Services/test_claims_review_metrics_scheduler.py
```

Expected: fail because current logs still interpolate raw exception text.

- [ ] **Step 3: Patch only the covered log branches**

Preserve:

- lookback parsing fallback behavior
- PostgreSQL vs SQLite path
- return counts
- scheduler startup behavior

- [ ] **Step 4: Run focused green**

Run the same command as Step 2.

- [ ] **Step 5: Run adjacent startup truthiness coverage**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/tests/Services/test_service_startup_truthiness_batch2.py \
  -k "CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED"
```

- [ ] **Step 6: Run Bandit and hand back**

Run Bandit on `claims_review_metrics_scheduler.py`, then return the same handoff package as Lane A.

---

### Task 6: Lane D Scout The Next Small-Service Batch

**Files:**
- Read-only: `tldw_Server_API/app/services/kanban_activity_cleanup_service.py`
- Read-only: `tldw_Server_API/app/services/workflows_webhook_dlq_service.py`
- Read-only: `tldw_Server_API/app/services/storage_cleanup_service.py`
- Read-only: `tldw_Server_API/tests/Services/test_service_startup_truthiness_batch2.py`
- Modify later by parent only: `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md`

- [ ] **Step 1: Inspect the first scout batch for direct-test cost**

For each candidate file, record:

- approximate file size / complexity
- whether an existing focused test file already exists
- whether a direct logger-stub test file can be added cheaply
- whether the remaining leak is a true Phase 3.3 fallback/log branch or a success-path/public-contract question

- [ ] **Step 2: Produce explicit lane decisions**

Use this format:

```text
approve now
source file: ...
exact branches: ...
existing tests reviewed: ...
why conservative: ...
```

or:

```text
defer
source file: ...
exact branches: ...
why not yet: ...
```

- [ ] **Step 3: Prefer `kanban_activity_cleanup_service.py` first if still narrow**

Reason: it is small, already adjacent to the scheduler family, and startup truthiness coverage already exists.

- [ ] **Step 4: Reject giant or policy-heavy candidates**

Examples likely to reject or defer:

- broad `storage_cleanup_service.py` if the branches touch too many public/runtime behaviors at once
- any file requiring shared-helper edits that would collide with active implementation lanes

- [ ] **Step 5: Hand off only the scout artifact**

The scout does not edit the shared phase plan. The parent copies `defer/reject` decisions into the plan before reusing Lane D.

---

### Task 7: Parent Merge Wave 1

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md`
- Integrate source/test files returned by Tasks 3-5

- [ ] **Step 1: Verify lane ownership stayed clean**

Reject or split any shard that touched files outside its approved source/test ownership.

- [ ] **Step 2: Merge only completed, independently green shards**

If one lane is not ready, do not block the others. Merge ready shards and leave the incomplete lane for the next wave.
Apply each ready lane back into the shared worktree using one explicit mechanism per shard: `git cherry-pick` for isolated lane commits, `apply_patch` for reviewed patch bundles, or tightly scoped manual copy for tiny diffs. Do not switch mechanisms mid-shard.

- [ ] **Step 3: Run the merged-wave pytest sweep**

Run the union of all touched test files from Wave 1, for example:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/tests/MediaDB2/test_sync_endpoint_errors.py \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/tests/Services/test_claims_alerts_scheduler.py \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/tests/Services/test_claims_review_metrics_scheduler.py \
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption/tldw_Server_API/tests/Services/test_service_startup_truthiness_batch2.py
```

- [ ] **Step 4: Run merged-wave Bandit**

Run Bandit across the touched Wave 1 source files only.

- [ ] **Step 5: Run hygiene checks**

Run:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption diff --check
git -C /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption status --short --branch
```

- [ ] **Step 6: Update the Phase 3.3 plan with merged-wave results**

Include:

- which lanes merged
- which scout candidates were deferred or rejected
- pytest and Bandit outcomes
- next approved candidate for the recycled scout lane

- [ ] **Step 7: Commit the merged wave**

Example:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_server2/.claude/worktrees/phase3.3-error-handler-adoption commit -m "Phase 3.3: parallel sync and claims scheduler sanitizers"
```

---

### Task 8: Repeat Wave Mechanics Until Exhaustion

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md`

- [ ] **Step 1: Reassign Lane D to the next scout-approved candidate**

Likely first follow-up candidate:

- `tldw_Server_API/app/services/kanban_activity_cleanup_service.py`

- [ ] **Step 2: Keep `sync.py` isolated if it was not merged in Wave 1**

Do not let unfinished `sync.py` work merge with a new medium endpoint tranche.

- [ ] **Step 3: Reuse the full Task 7 verification and commit sequence for every recycled wave**

For every later wave, repeat all of Task 7:

- ownership check
- parent merge
- merged-wave pytest sweep
- merged-wave Bandit
- hygiene checks
- Phase 3.3 plan update
- one logical commit

- [ ] **Step 4: Stop when only deferred/rejected candidates remain**

The parent stop condition is:

- no scout-approved candidates left in the audit list
- all deferred items explicitly recorded in the Phase 3.3 plan
- remaining files are policy-heavy, success-path, giant, or insufficiently covered

- [ ] **Step 5: Produce a final Phase 3.3 remainder summary**

List:

- completed tranches
- deferred items
- rejected items
- reasons each remainder stayed out of Phase 3.3
