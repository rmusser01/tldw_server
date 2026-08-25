# Durable Jobs Idempotency Receipts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one backend-neutral Jobs primitive that atomically admits owner-scoped user operations, durably replays them after archival, and prevents concurrent duplicate work.

**Architecture:** Add a small receipt table beside `jobs` and `jobs_archive`; it stores only immutable request correlation and points at one immutable Job UUID. Extend the existing typed Jobs operation layer with one command/result contract and matching SQLite/PostgreSQL admission implementations, then expose it through `JobManager` with one consistent active-plus-archive UUID read. Receipt-backed terminal Jobs are always archived before pruning, and expired terminal receipts are pruned separately after a minimum 30-day replay window.

**Tech Stack:** Python 3.10+, dataclasses, SQLite, PostgreSQL/psycopg, pytest, Hypothesis where useful, Bandit.

**Spec:** `Docs/superpowers/specs/2026-08-25-shared-workspace-clone-jobs-design.md`

## Global Constraints

- Jobs is the only source of operation status, progress, result, and error; receipts contain correlation and fingerprint metadata only.
- Raw client idempotency keys are never persisted; callers provide a bounded digest.
- Receipt replay is owner-scoped and survives active-to-archive movement for at least 30 days.
- Missing, duplicate, malformed, or cross-owner receipt correlations fail closed and never create replacement work.
- SQLite and PostgreSQL implement the same outcomes and transaction boundaries.
- No clone API, worker lifecycle, Workspace copy, or frontend behavior is implemented in this task.

---

## File Structure

- `tldw_Server_API/app/core/Jobs/operations/contracts.py`: backend-neutral command, disposition, result, and typed conflict/unavailable errors.
- `tldw_Server_API/app/core/Jobs/operations/sqlite/idempotency.py`: SQLite receipt lookup, owner/scope convergence, atomic Job-plus-receipt admission, and consistent UUID read.
- `tldw_Server_API/app/core/Jobs/operations/postgres/idempotency.py`: PostgreSQL parity using transaction-scoped advisory locking and row locking.
- `tldw_Server_API/app/core/Jobs/operations/sqlite/__init__.py`: export the SQLite operations.
- `tldw_Server_API/app/core/Jobs/operations/postgres/__init__.py`: export the PostgreSQL operations.
- `tldw_Server_API/app/core/Jobs/migrations.py`: SQLite receipt DDL, indexes, and forward migration checks.
- `tldw_Server_API/app/core/Jobs/pg_migrations.py`: PostgreSQL receipt DDL, indexes, grants, and RLS policy.
- `tldw_Server_API/app/core/Jobs/manager.py`: validation, backend dispatch, consistent operation lookup, retention, and receipt-aware pruning.
- `tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py`: SQLite behavior and concurrency tests.
- `tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_postgres.py`: PostgreSQL parity and concurrency tests.
- `tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py`: SQLite schema contract.
- `tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py`: PostgreSQL schema/RLS contract.
- `tldw_Server_API/tests/Jobs/test_jobs_prune_sqlite.py`: receipt-backed archive and retention behavior.
- `tldw_Server_API/tests/Jobs/test_jobs_prune_postgres.py`: PostgreSQL prune parity.
- `.github/workflows/ci.yml`: add the two new test modules to the existing Jobs shards if the coverage guard reports them uncovered.

### Task 1: Define The Backend-Neutral Receipt Contract

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/operations/contracts.py`
- Test: `tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py`

**Interfaces:**
- Consumes: existing `CreateJobCommand` and normalized Job row dictionaries.
- Produces: `IdempotentOperationCommand`, `IdempotentOperationDisposition`, `IdempotentOperationAdmission`, `IdempotentOperationConflict`, and `IdempotentOperationUnavailableError`.

- [x] **Step 1: Write contract invariant tests**

```python
def test_idempotent_operation_command_requires_owner_and_bounded_digests():
    with pytest.raises(ValueError, match="owner_user_id"):
        IdempotentOperationCommand(
            job=CreateJobCommand("sharing", "workspace-clone", "workspace_clone", {}, None),
            key_digest="a" * 64,
            request_fingerprint="b" * 64,
            operation_scope="share:1",
            receipt_expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )

def test_idempotent_admission_freezes_job_row():
    row = {"uuid": "job-1", "status": "queued"}
    result = IdempotentOperationAdmission.created(row)
    row["status"] = "failed"
    assert result.job["status"] == "queued"
```

- [x] **Step 2: Run the contract tests and verify the imports fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py -k contract -v`

Expected: FAIL because the new contracts do not exist.

- [x] **Step 3: Add immutable contracts with explicit outcomes**

```python
class IdempotentOperationDisposition(str, Enum):
    CREATED = "created"
    REPLAYED = "replayed"
    CONVERGED = "converged"

class IdempotentOperationConflictReason(str, Enum):
    KEY_REUSED = "idempotency_key_reused"
    SCOPE_ACTIVE = "operation_already_in_progress"

@dataclass(frozen=True)
class IdempotentOperationCommand:
    job: CreateJobCommand
    key_digest: str
    request_fingerprint: str
    operation_scope: str
    receipt_expires_at: datetime

@dataclass(frozen=True)
class IdempotentOperationAdmission:
    job: dict[str, Any]
    disposition: IdempotentOperationDisposition

class IdempotentOperationConflict(RuntimeError):
    def __init__(self, reason: IdempotentOperationConflictReason, job_uuid: str | None = None) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.job_uuid = job_uuid

class IdempotentOperationUnavailableError(RuntimeError):
    """The receipt-to-Job correlation cannot be proven safe."""
```

Validate ASCII bounded fields, exactly 64 lowercase hexadecimal characters for both digests, a non-empty owner, an operation scope of at most 200 ASCII characters, and a timezone-aware expiry at least 30 days after admission time. Copy mutable row data in result construction.

- [x] **Step 4: Run contract tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py -k contract -v`

Expected: PASS.

- [x] **Step 5: Commit the contract**

```bash
git add tldw_Server_API/app/core/Jobs/operations/contracts.py tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py
git commit -m "feat(jobs): define durable idempotency receipt contract"
```

### Task 2: Add SQLite And PostgreSQL Receipt Schema Parity

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/migrations.py`
- Modify: `tldw_Server_API/app/core/Jobs/pg_migrations.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py`

**Interfaces:**
- Consumes: validated digest, fingerprint, scope, owner, and immutable Job identity from Task 1.
- Produces: `job_idempotency_receipts` with identical logical columns and constraints on both backends.

- [ ] **Step 1: Add failing SQLite and PostgreSQL schema assertions**

Assert these columns exist: `receipt_id`, `domain`, `queue`, `job_type`, `owner_user_id`, `key_digest`, `request_fingerprint`, `operation_scope`, `job_uuid`, `job_id`, `created_at`, `expires_at`. Assert a unique index over `(domain, queue, job_type, owner_user_id, key_digest)`, lookup indexes on `job_uuid` and `(operation_scope, owner_user_id, expires_at)`, and no column capable of storing a raw client key.

- [ ] **Step 2: Run migration tests and verify failure**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py -k idempotency_receipt -v`

Expected: FAIL because the table and policies do not exist.

- [ ] **Step 3: Add backend-parity DDL**

Use the following logical schema in both migration modules, translating identity and timestamp types for each backend:

```sql
CREATE TABLE IF NOT EXISTS job_idempotency_receipts (
  receipt_id INTEGER PRIMARY KEY,
  domain TEXT NOT NULL,
  queue TEXT NOT NULL,
  job_type TEXT NOT NULL,
  owner_user_id TEXT NOT NULL,
  key_digest TEXT NOT NULL,
  request_fingerprint TEXT NOT NULL,
  operation_scope TEXT NOT NULL,
  job_uuid TEXT NOT NULL,
  job_id INTEGER NOT NULL,
  created_at TIMESTAMP NOT NULL,
  expires_at TIMESTAMP NOT NULL,
  UNIQUE(domain, queue, job_type, owner_user_id, key_digest)
);
```

Do not add a foreign key to `jobs`: the referenced active row intentionally moves to `jobs_archive`. Add check constraints for digest lengths and bounded text where PostgreSQL supports them; manager validation remains authoritative on both backends.

- [ ] **Step 4: Add PostgreSQL grants and RLS**

Enable and force RLS with the same `domain_filter` and `owner_filter` used by `jobs`. Add `job_idempotency_receipts_select` and `job_idempotency_receipts_modify` policies, sequence usage grants, and include the table in the migration debug inventory.

- [ ] **Step 5: Run schema and compatibility tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_sqlite.py -v`

Run when PostgreSQL fixture is available: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_postgres.py -v`

Expected: PASS; PostgreSQL may skip only through the canonical unavailable fixture.

- [ ] **Step 6: Commit schema parity**

```bash
git add tldw_Server_API/app/core/Jobs/migrations.py tldw_Server_API/app/core/Jobs/pg_migrations.py tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py
git commit -m "feat(jobs): persist owner-scoped idempotency receipts"
```

### Task 3: Implement Atomic Receipt Admission On SQLite

**Files:**
- Create: `tldw_Server_API/app/core/Jobs/operations/sqlite/idempotency.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/__init__.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py`

**Interfaces:**
- Consumes: `IdempotentOperationCommand`.
- Produces: `JobManager.admit_idempotent_operation(command: IdempotentOperationCommand) -> IdempotentOperationAdmission`.

- [ ] **Step 1: Write failing admission and replay tests**

Cover: first request creates Job and receipt atomically; same key and fingerprint replays the same UUID; same key with another fingerprint raises `KEY_REUSED`; a second key with the same owner/scope/fingerprint creates a receipt alias and returns `CONVERGED`; a second key with a different fingerprint raises `SCOPE_ACTIVE`; a forced receipt insert failure leaves no Job row.

- [ ] **Step 2: Write a real SQLite concurrency test**

Use `ThreadPoolExecutor(max_workers=8)` with one `JobManager` per thread and a barrier. Submit the same owner/scope/fingerprint under two keys. Assert every successful result has one UUID, exactly one active Job exists, and exactly two receipt rows exist.

- [ ] **Step 3: Run tests and verify failure**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py -v`

Expected: FAIL because the admission method is absent.

- [ ] **Step 4: Implement one `BEGIN IMMEDIATE` admission transaction**

The backend function signature is:

Implement `admit_idempotent_operation(conn: sqlite3.Connection, *, command: IdempotentOperationCommand, uuid_value: str, now: datetime, counters_enabled: bool) -> IdempotentOperationAdmission`.

Within one write transaction: read the exact receipt; resolve and validate its Job UUID when present; otherwise find queued/processing Jobs by exact domain/queue/type/owner and `batch_group == operation_scope`; converge only an equal fingerprint encoded in the bounded Job payload; insert an alias receipt when converging; otherwise insert the Job, created event, counters, and receipt. Compare fingerprints with `secrets.compare_digest`. Any losing unique race must reread and validate the winning receipt before returning.

- [ ] **Step 5: Add manager validation and dispatch**

`JobManager.admit_idempotent_operation` applies queue policy and quota behavior consistently with `create_job`, generates one UUID, and dispatches to the backend implementation. Do not implement this by calling `create_job()` and then inserting a receipt, because that creates a crash window.

- [ ] **Step 6: Run SQLite tests and existing admission regressions**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_manager.py tldw_Server_API/tests/Jobs/test_jobs_finalize_idempotency_sqlite.py -v`

Expected: PASS.

- [ ] **Step 7: Commit SQLite admission**

```bash
git add tldw_Server_API/app/core/Jobs/operations/sqlite/idempotency.py tldw_Server_API/app/core/Jobs/operations/sqlite/__init__.py tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py
git commit -m "feat(jobs): admit receipt-backed operations atomically on sqlite"
```

### Task 4: Add PostgreSQL Admission Parity

**Files:**
- Create: `tldw_Server_API/app/core/Jobs/operations/postgres/idempotency.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/__init__.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_postgres.py`

**Interfaces:**
- Consumes: the same command and result contracts as SQLite.
- Produces: identical outcomes through the same `JobManager` method.

- [ ] **Step 1: Port the behavior matrix as PostgreSQL tests**

Use the canonical `jobs_pg_dsn` fixture. Add exact replay, key mismatch, same-scope convergence, conflicting-scope request, rollback, and owner-isolation tests.

- [ ] **Step 2: Add a PostgreSQL concurrency test**

Start independent connections behind a barrier. Assert the transaction-scoped advisory lock yields one Job UUID and deterministic alias receipts without deadlocks or leaked transactions.

- [ ] **Step 3: Run tests and verify failure**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_postgres.py -v`

Expected: FAIL because PostgreSQL dispatch is not implemented.

- [ ] **Step 4: Implement advisory-locked admission**

The backend function signature mirrors SQLite and additionally receives `cursor_factory` and `advisory_xact_lock_key`. Acquire `pg_advisory_xact_lock` before receipt/scope reads, use `FOR KEY SHARE` for existing Job validation, and insert the Job/event/receipt in one connection transaction. Map unique violations only after reading and validating the winning receipt; propagate other database errors through the Jobs manager's existing backend error boundary.

- [ ] **Step 5: Run parity and existing PostgreSQL admission tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_postgres.py tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_postgres.py tldw_Server_API/tests/Jobs/test_jobs_pg_single_update_acquire_toggle.py -v`

Expected: PASS or canonical fixture skip.

- [ ] **Step 6: Commit PostgreSQL parity**

```bash
git add tldw_Server_API/app/core/Jobs/operations/postgres/idempotency.py tldw_Server_API/app/core/Jobs/operations/postgres/__init__.py tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_postgres.py
git commit -m "feat(jobs): add postgres receipt admission parity"
```

### Task 5: Add Consistent Active-Archive UUID Replay

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/idempotency.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/idempotency.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_postgres.py`

**Interfaces:**
- Consumes: immutable Job UUID plus optional domain and owner constraints.
- Produces: `JobManager.get_job_or_archived_by_uuid(job_uuid: str, *, domain: str | None = None, owner_user_id: str | None = None) -> dict[str, Any] | None`.

- [ ] **Step 1: Write failing active/archive/corruption tests**

Assert active and archived rows normalize identically except `archived`; moving a Job during lookup cannot return a false missing result; duplicate UUIDs across active/archive, a malformed UUID, receipt owner mismatch, or receipt Job ID/UUID mismatch raises `IdempotentOperationUnavailableError`.

- [ ] **Step 2: Run the tests and verify failure**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_postgres.py -k 'archive or corrupt or uuid' -v`

- [ ] **Step 3: Implement one consistent read per backend**

Use one connection and one transaction snapshot. Query active and archive candidates by UUID with owner/domain filters, normalize payload/result through existing helpers, require zero or exactly one candidate, and set `archived`. Do not compose `get_job_by_uuid()` and `get_job_or_archived()`, because those methods open separate connections and retain the archival race gap.

- [ ] **Step 4: Use the consistent lookup for receipt replay**

Receipt replay validates `job_uuid`, `job_id`, domain, queue, type, and owner against the resolved Job. A missing or ambiguous match is unavailable, not a new admission.

- [ ] **Step 5: Run focused replay tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_postgres.py -v`

- [ ] **Step 6: Commit consistent replay**

```bash
git add tldw_Server_API/app/core/Jobs/operations tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_postgres.py
git commit -m "fix(jobs): make receipt replay archive-consistent"
```

### Task 6: Enforce Receipt-Aware Archive And Retention

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_prune_sqlite.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_prune_postgres.py`

**Interfaces:**
- Consumes: terminal receipt-backed Jobs and receipt expiry timestamps.
- Produces: `JobManager.prune_idempotency_receipts(*, now: datetime | None = None, limit: int = 1000) -> int` and receipt-safe `prune_jobs()`.

- [ ] **Step 1: Write failing prune and retention tests**

Cover archive-before-delete disabled, mixed receipt/non-receipt batches, nonterminal expired receipt preservation, terminal unexpired replay, terminal expired receipt deletion only after the Job is archived, and idempotent repeated pruning.

- [ ] **Step 2: Run prune tests and verify failure**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_prune_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_prune_postgres.py -k receipt -v`

- [ ] **Step 3: Archive receipt-backed candidates unconditionally**

Within each existing prune transaction, partition locked candidates into globally archived rows and receipt-backed rows. Insert every receipt-backed row into `jobs_archive` before deleting active rows, using the existing exact archive projection and UUID collision checks. Keep non-receipt behavior controlled by `JOBS_ARCHIVE_BEFORE_DELETE`.

- [ ] **Step 4: Implement bounded receipt pruning**

Delete at most `limit` receipts whose `expires_at <= now`, whose referenced Job is terminal, and whose UUID exists exactly once in `jobs_archive`. Never delete a receipt referencing active/nonterminal work. Keep this operation separate from `prune_jobs` so routine Job pruning does not silently shorten the 30-day replay contract.

- [ ] **Step 5: Run all Jobs prune tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_prune_sqlite.py tldw_Server_API/tests/Jobs/test_prune_jobs.py tldw_Server_API/tests/Jobs/test_jobs_prune_postgres.py -v`

- [ ] **Step 6: Commit retention behavior**

```bash
git add tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/tests/Jobs/test_jobs_prune_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_prune_postgres.py
git commit -m "fix(jobs): preserve receipt-backed operations through pruning"
```

### Task 7: Complete Parity, Security, And CI Verification

**Files:**
- Modify if required: `.github/workflows/ci.yml`
- Modify: `backlog/tasks/task-12020.46 - Add-durable-Jobs-idempotency-receipts-for-user-operations.md`

**Interfaces:**
- Consumes: all Task 1-6 deliverables.
- Produces: a reviewable foundation and exact verification evidence for `TASK-12020.48`.

- [ ] **Step 1: Run the focused Jobs suite**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_prune_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_postgres.py tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py tldw_Server_API/tests/Jobs/test_jobs_prune_postgres.py -v`

- [ ] **Step 2: Run static quality checks**

Run: `source .venv/bin/activate && python -m ruff check tldw_Server_API/app/core/Jobs/operations tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/app/core/Jobs/migrations.py tldw_Server_API/app/core/Jobs/pg_migrations.py tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_postgres.py`

Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Jobs/operations tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/app/core/Jobs/migrations.py tldw_Server_API/app/core/Jobs/pg_migrations.py -f json -o /tmp/bandit_task_12020_46.json`

- [ ] **Step 3: Verify CI shard ownership**

Run: `source .venv/bin/activate && python Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml`

If the two new test modules are uncovered, add them to the existing Jobs SQLite/PostgreSQL shard groups and rerun until the guard passes.

- [ ] **Step 4: Run diff checks and inspect the patch**

Run: `git diff --check`

Run: `git diff --stat origin/dev...HEAD && git status --short`

Confirm the pre-existing untracked `apps/packages/ui/node_modules` symlink is neither staged nor modified.

- [ ] **Step 5: Update Backlog and commit closeout**

Record exact test results, PostgreSQL fixture availability, Bandit output path, touched files, and residual risks in `TASK-12020.46`; check acceptance criteria only with evidence.

```bash
git add .github/workflows/ci.yml backlog/tasks/task-12020.46\ -\ Add-durable-Jobs-idempotency-receipts-for-user-operations.md
git commit -m "chore(jobs): close durable receipt foundation"
```

## Self-Review

- Spec coverage: schema parity, atomic admission, owner/scope locking, archived replay, fail-closed correlation, RLS, 30-day retention, receipt-aware pruning, and CI ownership each map to a task above.
- Placeholder scan: no deferred implementation steps or undefined helper references remain.
- Type consistency: Tasks 3-6 consume the exact Task 1 command/result/error names; `JobManager.admit_idempotent_operation` and `get_job_or_archived_by_uuid` are the only public foundation methods consumed by `TASK-12020.48`.
- Scope control: clone-specific request validation, API error mapping, worker registration, and UI persistence remain outside this plan.
