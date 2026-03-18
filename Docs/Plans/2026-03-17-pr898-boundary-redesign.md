# PR 898 Jobs And Metering Boundary Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** Complete

## Execution Summary

Completed on `2026-03-17` in worktree `codex-pr898-boundary-redesign`.

Implemented commits:

- `de8569d55` `refactor: add jobs repository boundary`
- `217722302` `refactor: route fair share through jobs repository`
- `126e71d31` `refactor: add authnz metering repositories`
- `71fe7f9a8` `refactor: split stripe metering orchestration from persistence`

Final verification:

- `python -m pytest tldw_Server_API/tests/test_stripe_metering.py tldw_Server_API/tests/Billing/test_authnz_metering_repository.py -v`
  - Result: `28 passed`
- `python -m pytest tldw_Server_API/tests/AuthNZ/test_consent_endpoints.py tldw_Server_API/tests/AuthNZ/test_audit_chain_integration.py tldw_Server_API/tests/Billing/test_overage_enforcement_integration.py tldw_Server_API/tests/Billing/test_authnz_metering_repository.py tldw_Server_API/tests/Jobs/test_fair_share_integration.py tldw_Server_API/tests/Jobs/test_jobs_repository.py tldw_Server_API/tests/test_stripe_metering.py -v`
  - Result: `67 passed`
- `python -m bandit -r tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/app/core/DB_Management/Jobs_Repository.py tldw_Server_API/app/core/DB_Management/AuthNZ_Metering_Repository.py tldw_Server_API/app/services/stripe_metering_service.py -f json -o /tmp/bandit_pr898_boundary_redesign.json`
  - Result: `0` findings, `0` errors

**Goal:** Redesign Jobs and Stripe metering persistence boundaries so `JobManager` and the Stripe metering service become orchestration façades backed by explicit repository/session layers.

**Architecture:** Introduce a dedicated Jobs repository/session layer under `app/core/DB_Management`, plus AuthNZ-backed repositories for metering usage, subscriptions, and sync-log persistence. Refactor `JobManager` and Stripe metering to depend on those layers instead of embedding SQL and schema bootstrap logic directly.

**Naming note:** This plan keeps the existing repository conventions under `app/core/DB_Management`, including the current package and module naming style already used throughout the codebase, rather than introducing a one-off naming scheme for only this refactor.

**Tech Stack:** Python 3.11, FastAPI, SQLite, PostgreSQL/asyncpg, psycopg, pytest, loguru, Bandit

---

### Task 1: Add Jobs Repository Scaffolding

**Status:** Complete

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/Jobs_Repository.py`
- Test: `tldw_Server_API/tests/Jobs/test_jobs_repository.py`
- Reference: `tldw_Server_API/app/core/Jobs/manager.py`

**Step 1: Write the failing tests**

Add tests that describe the new persistence API:

```python
def test_count_active_jobs_for_user_sqlite_uses_session_connection():
    repo = JobsRepository.for_sqlite(db_path)
    with repo.session() as session:
        assert repo.count_active_jobs_for_user("42", session=session) == 2
```

```python
def test_create_job_session_can_read_and_write_in_one_transaction():
    repo = JobsRepository.for_sqlite(db_path)
    with repo.session() as session:
        active = repo.count_active_jobs_for_user("42", session=session)
        row = repo.insert_job(..., session=session)
        assert active == 0
        assert row["owner_user_id"] == "42"
```

**Step 2: Run the new tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Jobs/test_jobs_repository.py -v
```

Expected: failures because `JobsRepository` and its session API do not exist yet.

**Step 3: Write the minimal implementation**

Add:

- `JobsSession` wrapper for live SQLite/psycopg connections
- `JobsRepository.session()`
- `JobsRepository.count_active_jobs_for_user(...)`
- `JobsRepository.insert_job(...)` only as far as needed by the tests

Keep backend-specific SQL inside this file only.

**Step 4: Run the repository tests again**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Jobs/test_jobs_repository.py -v
```

Expected: pass.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/Jobs_Repository.py tldw_Server_API/tests/Jobs/test_jobs_repository.py
git commit -m "refactor: add jobs repository boundary"
```

### Task 2: Refactor JobManager To Use The Repository Boundary

**Status:** Complete

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Test: `tldw_Server_API/tests/Jobs/test_fair_share_integration.py`
- Test: `tldw_Server_API/tests/Jobs/test_jobs_repository.py`

**Step 1: Write the failing test**

Add a test that proves `create_job()` uses repository-backed counting instead of opening a second connection:

```python
def test_create_job_reuses_repository_session_for_fair_share(monkeypatch):
    repo = FakeJobsRepository(...)
    jm = JobManager(db_path, jobs_repository=repo)
    jm.create_job(...)
    assert repo.count_calls == 1
    assert repo.insert_used_same_session is True
```

**Step 2: Run the focused tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Jobs/test_fair_share_integration.py tldw_Server_API/tests/Jobs/test_jobs_repository.py -v
```

Expected: failure because `JobManager` does not accept/use the repository yet.

**Step 3: Write the minimal implementation**

Refactor `JobManager` so that:

- constructor accepts optional `jobs_repository`
- default construction builds the repository from existing config
- `_count_active_jobs_for_user()` delegates to repository/session behavior or is removed entirely
- `create_job()` opens a repository session, performs fair-share counting there, then inserts through the repository

Do not change the public `create_job()` call signature.

**Step 4: Run the Jobs tests again**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Jobs/test_fair_share_integration.py tldw_Server_API/tests/Jobs/test_jobs_repository.py -v
```

Expected: pass.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/tests/Jobs/test_fair_share_integration.py tldw_Server_API/tests/Jobs/test_jobs_repository.py
git commit -m "refactor: route fair share through jobs repository"
```

### Task 3: Add Metering Repositories

**Status:** Complete

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/AuthNZ_Metering_Repository.py`
- Test: `tldw_Server_API/tests/Billing/test_authnz_metering_repository.py`
- Reference: `tldw_Server_API/app/services/stripe_metering_service.py`

**Step 1: Write the failing tests**

Cover the repository contract:

```python
async def test_usage_repository_normalizes_legacy_rows():
    repo = AuthNZUsageDailyRepository(pool=fake_pool)
    rows = await repo.fetch_usage_for_date("2026-03-13")
    assert rows[0]["bytes_in_total"] == 0
```

```python
async def test_subscription_repository_falls_back_to_org_owner():
    repo = AuthNZBillingSubscriptionRepository(pool=fake_pool)
    row = await repo.get_active_subscription_for_user(42)
    assert row["stripe_subscription_id"] == "sub_123"
```

```python
async def test_sync_log_repository_ensures_and_records_schema():
    repo = AuthNZMeteringSyncLogRepository(pool=fake_pool)
    await repo.ensure_schema()
    await repo.record_sync(...)
```

**Step 2: Run the repository tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Billing/test_authnz_metering_repository.py -v
```

Expected: failures because the repositories do not exist yet.

**Step 3: Write the minimal implementation**

Add repository classes for:

- usage reads
- subscription lookup
- sync-log schema/bootstrap and sync state

Keep all SQL and DDL in this module. Normalize SQLite/PostgreSQL row shapes before returning. Keeping the three metering repositories in one boundary module is intentional for this pass because they share the same AuthNZ pool semantics and form one cohesive metering persistence surface.

**Step 4: Run the repository tests again**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Billing/test_authnz_metering_repository.py -v
```

Expected: pass.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/AuthNZ_Metering_Repository.py tldw_Server_API/tests/Billing/test_authnz_metering_repository.py
git commit -m "refactor: add authnz metering repositories"
```

### Task 4: Refactor Stripe Metering Into Orchestration Plus Repositories

**Status:** Complete

**Files:**
- Modify: `tldw_Server_API/app/services/stripe_metering_service.py`
- Modify: `tldw_Server_API/tests/test_stripe_metering.py`
- Test: `tldw_Server_API/tests/Billing/test_authnz_metering_repository.py`

**Step 1: Write the failing service test**

Add a service-level test that injects fake repositories:

```python
async def test_sync_daily_usage_uses_repositories_and_records_sync():
    svc = StripeMeteringService(
        usage_repo=fake_usage_repo,
        subscription_repo=fake_subscription_repo,
        sync_log_repo=fake_sync_log_repo,
        stripe_client=fake_stripe_client,
    )
    result = await svc.sync_daily_usage(date="2026-03-13")
    assert result["synced_users"] == 1
```

**Step 2: Run the focused metering tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/test_stripe_metering.py tldw_Server_API/tests/Billing/test_authnz_metering_repository.py -v
```

Expected: failures because the service still depends on internal SQL helpers.

**Step 3: Write the minimal implementation**

Refactor `StripeMeteringService` so that:

- constructor accepts optional repositories and optional db-pool provider
- SQL helper methods are deleted or reduced to thin adapter creation
- `sync_daily_usage()` and `check_reconciliation()` operate through repository interfaces
- Stripe calls remain in the service layer

Preserve the external return payloads.

**Step 4: Run the metering tests again**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/test_stripe_metering.py tldw_Server_API/tests/Billing/test_authnz_metering_repository.py -v
```

Expected: pass.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/services/stripe_metering_service.py tldw_Server_API/tests/test_stripe_metering.py tldw_Server_API/tests/Billing/test_authnz_metering_repository.py
git commit -m "refactor: split stripe metering orchestration from persistence"
```

### Task 5: Run Full Verification And Update PR Threading

**Status:** Complete

**Files:**
- Modify: `Docs/Plans/2026-03-17-pr898-boundary-redesign-design.md`
- Modify: `Docs/Plans/2026-03-17-pr898-boundary-redesign.md`
- Review: `tldw_Server_API/app/core/Jobs/manager.py`
- Review: `tldw_Server_API/app/core/DB_Management/Jobs_Repository.py`
- Review: `tldw_Server_API/app/core/DB_Management/AuthNZ_Metering_Repository.py`
- Review: `tldw_Server_API/app/services/stripe_metering_service.py`

**Step 1: Run the expanded regression suite**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/test_consent_endpoints.py \
  tldw_Server_API/tests/AuthNZ/test_audit_chain_integration.py \
  tldw_Server_API/tests/Billing/test_overage_enforcement_integration.py \
  tldw_Server_API/tests/Billing/test_authnz_metering_repository.py \
  tldw_Server_API/tests/Jobs/test_fair_share_integration.py \
  tldw_Server_API/tests/Jobs/test_jobs_repository.py \
  tldw_Server_API/tests/test_stripe_metering.py -v
```

Expected: all pass.

**Step 2: Run Bandit on touched scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/DB_Management/Jobs_Repository.py \
  tldw_Server_API/app/core/DB_Management/AuthNZ_Metering_Repository.py \
  tldw_Server_API/app/services/stripe_metering_service.py \
  -f json -o /tmp/bandit_pr898_boundary_redesign.json
```

Expected: `0` new findings in touched files.

**Step 3: Update PR materials**

- Refresh the design/plan docs if the final implementation deviated.
- Update the draft PR body with the final architectural summary and verification output.
- Reply on the follow-up PR threads with the repository-boundary changes.

**Step 4: Commit**

```bash
git add Docs/Plans/2026-03-17-pr898-boundary-redesign-design.md Docs/Plans/2026-03-17-pr898-boundary-redesign.md
git commit -m "docs: finalize boundary redesign execution notes"
```

Plan complete and execution notes are now finalized in this file.
