# PR 908 Open Review Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** Complete

**Goal:** Close the remaining PR 908 review threads by fixing the Postgres fair-share/RLS ordering bug and adding the missing repository documentation and test type hints.

**Architecture:** Keep the existing Jobs and metering repository redesign intact. Adjust `JobManager.create_job()` so the Postgres fair-share count runs only after `_pg_cursor()` applies RLS state, and document the new repository boundaries directly in the new modules.

**Tech Stack:** Python 3.11, FastAPI, SQLite, PostgreSQL/psycopg, pytest, loguru, Bandit

---

### Task 1: Add The Failing Fair-Share Ordering Regression

**Files:**
- Modify: `tldw_Server_API/tests/Jobs/test_fair_share_integration.py`
- Reference: `tldw_Server_API/app/core/Jobs/manager.py`

**Step 1: Write the failing test**

Add a focused regression test that exercises the Postgres create path and proves fair-share counting does not happen until after `_pg_cursor()` has been entered.

```python
def test_postgres_fair_share_count_runs_after_pg_cursor_setup(...):
    events = []
    repo = TrackingJobsRepository(...)
    manager = JobManager(..., jobs_repository=repo)
    ...
    assert events.index("pg_cursor_enter") < events.index("count_active_jobs")
```

Also add explicit parameter and return annotations to the `TrackingJobsRepository` helper methods in the same file.

**Step 2: Run the focused test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Jobs/test_fair_share_integration.py -k postgres_fair_share_count_runs_after_pg_cursor_setup -v
```

Expected: the new test fails because the current Postgres branch counts before `_pg_cursor()`.

**Step 3: Commit checkpoint**

Do not commit yet. Keep this red test in place while implementing the fix.

### Task 2: Move Postgres Fair-Share Counting Behind `_pg_cursor()`

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Reference: `tldw_Server_API/app/core/DB_Management/Jobs_Repository.py`

**Step 1: Write the minimal implementation**

Refactor `create_job()` so that:

- SQLite keeps the current early fair-share block.
- PostgreSQL skips the early fair-share block.
- Inside `with conn: with self._pg_cursor(conn) as cur:` the Postgres branch performs the fair-share check before quota, idempotency, and insert logic.
- The count and insert reuse the same `JobsSession`.

Use `BadRequestError` for the admission-control rejection to match the established narrow-scope cleanup already applied elsewhere in Jobs.

**Step 2: Run the focused Jobs tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Jobs/test_fair_share_integration.py tldw_Server_API/tests/Jobs/test_jobs_repository.py -v
```

Expected: the new regression and the existing Jobs tests pass.

**Step 3: Commit**

```bash
git add tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/tests/Jobs/test_fair_share_integration.py
git commit -m "fix: restore postgres fair-share rls ordering"
```

### Task 3: Add Repository Boundary Docstrings

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Jobs_Repository.py`
- Modify: `tldw_Server_API/app/core/DB_Management/AuthNZ_Metering_Repository.py`

**Step 1: Add the missing docstrings**

Document:

- module responsibilities
- class roles
- backend-specific session/transaction behavior
- method contracts and normalized return shapes

Keep the wording concise and behavior-focused.

**Step 2: Run the directly affected tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Billing/test_authnz_metering_repository.py tldw_Server_API/tests/test_stripe_metering.py -v
```

Expected: pass.

**Step 3: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/Jobs_Repository.py tldw_Server_API/app/core/DB_Management/AuthNZ_Metering_Repository.py
git commit -m "docs: document repository boundaries"
```

### Task 4: Full Verification And PR Cleanup

**Files:**
- Modify: `Docs/Plans/2026-03-17-pr908-open-review-cleanup-design.md`
- Modify: `Docs/Plans/2026-03-17-pr908-open-review-cleanup.md`

**Step 1: Run the scoped regression suite**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Billing/test_authnz_metering_repository.py \
  tldw_Server_API/tests/Jobs/test_fair_share_integration.py \
  tldw_Server_API/tests/Jobs/test_jobs_repository.py \
  tldw_Server_API/tests/test_stripe_metering.py -v
```

Expected: pass.

**Step 2: Run Bandit on the touched backend files**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/DB_Management/Jobs_Repository.py \
  tldw_Server_API/app/core/DB_Management/AuthNZ_Metering_Repository.py \
  -f json -o /tmp/bandit_pr908_open_review_cleanup.json
```

Expected: `0` findings and `0` errors in the JSON output.

**Step 3: Update plan docs with results**

Record the final verification output and any notable implementation notes in these two plan files.

**Step 4: Push and resolve review threads**

Run the normal git push, then reply in each PR 908 thread with the exact fix location and resolve the thread.

## Execution Summary

Completed on `2026-03-17` in worktree `codex-pr898-boundary-redesign`.

Implemented changes:

- moved Postgres fair-share evaluation behind `_pg_cursor()` while keeping the same `JobsSession`
- added a deterministic fake-Postgres regression for fair-share/RLS ordering
- added explicit type hints to `TrackingJobsRepository`
- documented the Jobs and AuthNZ metering repository boundaries with module/class/method docstrings

Final verification:

- `python -m pytest tldw_Server_API/tests/Billing/test_authnz_metering_repository.py tldw_Server_API/tests/Jobs/test_fair_share_integration.py tldw_Server_API/tests/Jobs/test_jobs_repository.py tldw_Server_API/tests/test_stripe_metering.py -v`
  - Result: `41 passed, 5 warnings`
- `python -m bandit -r tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/app/core/DB_Management/Jobs_Repository.py tldw_Server_API/app/core/DB_Management/AuthNZ_Metering_Repository.py -f json -o /tmp/bandit_pr908_open_review_cleanup.json`
  - Result: `0` findings, `0` errors
