# PR 908 Review Followups Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the remaining PR 908 review threads without broadening scope beyond the live unresolved comments.

**Architecture:** Keep the existing Jobs and metering repository boundaries intact while hardening their configuration and lifecycle behavior. Preserve the current repository/session split, but move shared exceptions into `app/core/exceptions.py`, stop stale AuthNZ pool caching, validate Jobs repository construction up front, and restore idempotent replay semantics when fair-share admission would otherwise reject a retry.

**Tech Stack:** Python, FastAPI backend modules, SQLite/PostgreSQL repository adapters, pytest, Bandit

---

### Task 1: Document the remaining review scope

**Files:**
- Create: `Docs/Plans/2026-03-17-pr908-review-followups.md`

**Step 1: Capture the live unresolved review threads**

Run:

```bash
gh api graphql -f query='query { repository(owner:"rmusser01", name:"tldw_server") { pullRequest(number: 908) { reviewThreads(first: 100) { nodes { id isResolved path line comments(first: 20) { nodes { id url body author { login } } } } } } } }'
```

Expected: unresolved threads limited to metering repository lifecycle, Jobs repository validation, and fair-share/idempotency behavior.

**Step 2: Record the execution plan**

Expected: this file exists and the remaining tasks below match the live thread set.

### Task 2: Write the failing tests first

**Files:**
- Modify: `tldw_Server_API/tests/Billing/test_authnz_metering_repository.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_repository.py`
- Modify: `tldw_Server_API/tests/Jobs/test_fair_share_integration.py`

**Step 1: Write the failing metering repository tests**

Add tests that:
- import `DuplicateActiveSubscriptionError` from `tldw_Server_API.app.core.exceptions`
- verify `_get_db_pool()` re-reads the global `get_db_pool()` when no pool was injected

**Step 2: Write the failing Jobs repository validation tests**

Add tests that:
- reject unsupported `backend`
- reject SQLite repositories without `db_path` when not pooled
- reject Postgres repositories without `db_url` when not pooled
- reject connection pools missing `acquire`

**Step 3: Write the failing fair-share/idempotency regression tests**

Add tests that prove:
- an idempotent retry returns the existing row when the user is already at the fair-share limit
- this holds for the SQLite create path

**Step 4: Run the new tests to verify they fail**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Billing/test_authnz_metering_repository.py tldw_Server_API/tests/Jobs/test_jobs_repository.py tldw_Server_API/tests/Jobs/test_fair_share_integration.py -q
```

Expected: failures in the new assertions before production code changes.

### Task 3: Implement the minimal fixes

**Files:**
- Modify: `tldw_Server_API/app/core/exceptions.py`
- Modify: `tldw_Server_API/app/core/DB_Management/AuthNZ_Metering_Repository.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Jobs_Repository.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`

**Step 1: Centralize the shared exceptions**

Move or define:
- `DuplicateActiveSubscriptionError`
- a repository configuration exception that subclasses the project exception hierarchy

**Step 2: Fix AuthNZ pool resolution**

Update `_get_db_pool()` so:
- injected pools are reused
- globally managed pools are fetched fresh from `get_db_pool()` instead of being cached on the repository instance

**Step 3: Harden `JobsRepository` construction**

Validate in `__init__`:
- backend is only `sqlite` or `postgres`
- SQLite requires `db_path` unless a pool is injected
- Postgres requires `db_url` unless a pool is injected
- invalid pool objects raise the project repository configuration exception

**Step 4: Restore idempotent retry semantics**

Update `JobManager.create_job()` so:
- existing idempotent rows are checked before fair-share rejection
- retries with the same idempotency key return the pre-existing row instead of raising `BadRequestError`
- the same repository session/transaction is preserved

### Task 4: Verify and close the review threads

**Files:**
- Modify: `tldw_Server_API/tests/Billing/test_authnz_metering_repository.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_repository.py`
- Modify: `tldw_Server_API/tests/Jobs/test_fair_share_integration.py`

**Step 1: Run the scoped regression suite**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Billing/test_authnz_metering_repository.py tldw_Server_API/tests/Jobs/test_jobs_repository.py tldw_Server_API/tests/Jobs/test_fair_share_integration.py tldw_Server_API/tests/test_stripe_metering.py -q
```

Expected: all selected tests pass.

**Step 2: Run Bandit on the touched production files**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/exceptions.py tldw_Server_API/app/core/DB_Management/AuthNZ_Metering_Repository.py tldw_Server_API/app/core/DB_Management/Jobs_Repository.py tldw_Server_API/app/core/Jobs/manager.py -f json -o /tmp/bandit_pr908_review_followups.json
```

Expected: `results=0` and `errors=0`.

**Step 3: Reply on GitHub and resolve threads**

Reply in-thread for:
- the five code fixes implemented here
- the one already-fixed `self._jobs_repository.session()` comment

Expected: PR 908 has zero unresolved review threads.
