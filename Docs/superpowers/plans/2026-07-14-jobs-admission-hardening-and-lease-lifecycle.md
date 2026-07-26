# Jobs Admission Hardening and Lease Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the three validated Jobs admission defects, then incrementally extract the single-job lease acquisition, renewal, and release transaction boundaries without changing the public `JobManager` API.

**Architecture:** Deliver three separate PRs. PR 1 keeps `JobManager.create_job` as the facade while making secret rejection authoritative, quota failures fail closed, optional PostgreSQL counters savepoint-isolated, and enabled quotas atomic. PostgreSQL uses owner/domain-scoped advisory locks; SQLite uses a short `BEGIN IMMEDIATE` transaction because SQLite write locking is database-wide. PR 2 starts only after PR 1 is merged and rebased; it adds a typed acquisition command and moves only single-job acquisition SQL into backend lifecycle modules. PR 3 starts only after PR 2 is merged and rebased; it adds typed renewal/release commands and moves those two transitions. Validation, compatibility mapping, and post-commit effects remain in `JobManager` throughout.

**Tech Stack:** Python 3.14, sqlite3, psycopg 3, PostgreSQL transaction-scoped advisory locks, SQLite `BEGIN IMMEDIATE`, dataclasses, pytest, existing Jobs temporary PostgreSQL fixtures, Loguru, Bandit.

## Global Constraints

- Parent tracking task: `TASK-12969`.
- Admission implementation task: `TASK-12969.1`.
- Lease acquisition task: `TASK-12969.2`, dependent on `TASK-12969.1`.
- Lease renewal/release task: `TASK-12969.3`, dependent on `TASK-12969.2`.
- Current execution base: `7c7d591c6e3552ca4bdbf30bdd6bf79460221ece` from `origin/dev`.
- Findings were reproduced on `132037dd075090c295003d6885ac4276a9640916`; the intervening upstream commits did not change Jobs source or tests, and each task reconfirms its red state before implementation.
- Preserve every public `JobManager` method signature and return shape.
- Backend operation modules must not import `JobManager`.
- Do not add or change database schema in any PR.
- Acquire quota locks only when the corresponding quota is enabled and an owner scope exists; do not globally serialize normal admissions or acquisitions.
- Durable job row, counter, and existing outbox writes remain in the backend transaction. Metrics, tracing, gauges, SLA reporting, and `emit_job_event` calls run after commit.
- `complete_job`, `fail_job`, `cancel_job`, terminal transitions, `batch_renew_leases`, batch completion/failure, retry, quarantine, pruning, and admin-owned SQL are out of scope.
- Real PostgreSQL tests use `pytest.mark.pg_jobs`, `jobs_pg_dsn`, and `RUN_JOBS=1`; a skipped PostgreSQL test is not acceptable evidence for any PR.
- Run Bandit on every touched production path before each PR is considered complete.
- Each PR requires a requester-owned `Change summary` explaining what changed and why these locking and transaction choices were used.

---

## Validated Baseline

The following regressions were reproduced against the base commit before this plan was written:

1. With `JOBS_SECRET_REJECT=true`, a payload containing `api_key` is persisted because the intentional `ValueError` is caught by `_JOB_NONCRITICAL_EXCEPTIONS`.
2. A PostgreSQL counter statement error is logged as noncritical, but the transaction remains aborted and the subsequent `job_events` insert raises `psycopg.errors.InFailedSqlTransaction`.
3. Two concurrent PostgreSQL submissions under `JOBS_QUOTA_MAX_QUEUED=1` both succeed because quota count and insert are not serialized.

Clean happy-path baseline rerun in the rebased isolated worktree:

```text
Focused SQLite/contracts/parity: 22 passed, 52 warnings
Real PostgreSQL parity:           7 passed, 20 warnings
```

The red diagnostic outcomes are expected until Tasks 1-4 are implemented:

```text
secret reject: DID NOT RAISE ValueError
counter failure: psycopg.errors.InFailedSqlTransaction
quota concurrency: 2 created, expected 1
```

## Stage Map

### Stage 1: Admission Policy and Transaction Recovery
**Goal:** Make secret rejection authoritative and isolate optional PostgreSQL counter failures.
**Success Criteria:** Secret-bearing jobs are not inserted in reject mode; a counter failure does not abort job/event commit.
**Tests:** Focused SQLite secret tests and real PostgreSQL counter fault injection.
**Status:** Complete

### Stage 2: Atomic Admission Quotas
**Goal:** Serialize quota check plus insert only within the enabled owner/domain quota scope and fail closed on quota query errors.
**Success Criteria:** Concurrent submissions cannot oversubscribe max-queued quotas; unrelated PostgreSQL scopes remain concurrent; quota-disabled SQLite admission retains its deferred transaction path.
**Tests:** Deterministic delayed-insert concurrency tests on SQLite and PostgreSQL plus existing quota/parity suites.
**Status:** Complete

### Stage 3: Acquisition Contract and Parity Safety Net
**Goal:** Define the typed acquisition command and characterize acquisition before moving SQL.
**Success Criteria:** Facade tests cover contention, expiry, dependencies, quotas, counters, ordering, and post-commit effects on both backends.
**Tests:** Contract tests, shared acquisition parity scenarios, direct backend operation tests.
**Status:** Not Started

### Stage 4: Acquisition Extraction
**Goal:** Move single-job acquisition SQL into backend modules and leave `JobManager.acquire_next_job` as a thin compatibility facade.
**Success Criteria:** SQLite and real PostgreSQL acquisition suites pass with no operation-module dependency on `JobManager`; renewal, release, batch, and terminal paths remain unchanged.
**Tests:** Focused acquisition matrix, acquisition concurrency stress, Jobs parity, Bandit, compile check.
**Status:** Not Started

### Stage 5: Renewal and Release Extraction
**Goal:** Characterize and move single-job renewal/release SQL only after acquisition is merged.
**Success Criteria:** Both backends preserve enforcement, no-shorten renewal, progress, field clearing, counters, and post-commit effects; batch and terminal paths remain unchanged.
**Tests:** Contract tests, shared renewal/release parity, direct operation tests, Jobs parity, Bandit, compile check.
**Status:** Not Started

## File Structure

### PR 1: Admission Hardening

- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
  - Separate secret scanner failures from intentional policy rejection.
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/admission.py`
  - Add a stable owner/domain quota lock key, transaction-scoped advisory locking, fail-closed quota queries, and counter savepoints.
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/admission.py`
  - Use `BEGIN IMMEDIATE` only for enabled owner-scoped quotas and fail closed on quota query errors.
- Create: `tldw_Server_API/tests/Jobs/test_jobs_secret_hygiene.py`
  - Cover reject, redact, scan-error fallback, and absence of inserts.
- Create: `tldw_Server_API/tests/Jobs/test_jobs_admission_operations_postgres.py`
  - Cover real counter failure recovery and quota-query exception policy.
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py`
  - Cover quota-query failure rollback.
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_quotas_postgres.py`
  - Cover deterministic concurrent max-queued admission and independent scope progress.
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_quotas_sqlite.py`
  - Cover deterministic concurrent max-queued admission without leaking `database is locked`.
- Modify if required by new cursor statements: `tldw_Server_API/tests/Jobs/test_jobs_manager_pg_fakes_extended.py`
  - Teach the fake cursor about advisory lock and savepoint statements without weakening assertions.
- Update through Backlog MCP: `TASK-12969` and `TASK-12969.1`.

### PR 2: Single-Job Lease Acquisition

- Modify: `tldw_Server_API/app/core/Jobs/operations/contracts.py`
  - Add the acquire command dataclass plus a precise no-eligible transition reason.
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/__init__.py`
  - Export SQLite acquisition.
- Create: `tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py`
  - Own the SQLite single-job acquisition transaction.
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/__init__.py`
  - Export PostgreSQL acquisition.
- Create: `tldw_Server_API/app/core/Jobs/operations/postgres/lifecycle.py`
  - Own the PostgreSQL single-job acquisition transaction.
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
  - Build acquisition commands, route by backend, map results, and run post-commit effects.
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py`
  - Test the acquisition command/result and retain the no-`JobManager` import guard.
- Create: `tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_sqlite.py`
  - Direct SQLite acquisition coverage.
- Create: `tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_postgres.py`
  - Direct real PostgreSQL acquisition coverage.
- Modify: `tldw_Server_API/tests/Jobs/parity/scenarios.py`
  - Add shared acquire contention and expired reclaim scenarios.
- Modify: `tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py`
  - Run new acquisition scenarios against SQLite.
- Modify: `tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py`
  - Run new acquisition scenarios against real PostgreSQL.
- Create: `tldw_Server_API/tests/Jobs/test_jobs_lifecycle_side_effects.py`
  - Prove acquisition in-process events and metrics happen once and only after an applied transition.
- Update through Backlog MCP: `TASK-12969` and `TASK-12969.2`.

### PR 3: Single-Job Lease Renewal and Release

- Modify: `tldw_Server_API/app/core/Jobs/operations/contracts.py`
  - Add renewal and release command dataclasses.
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/__init__.py`
  - Export SQLite renewal and release.
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py`
  - Own SQLite single-job renewal and release transactions.
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/__init__.py`
  - Export PostgreSQL renewal and release.
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/lifecycle.py`
  - Own PostgreSQL single-job renewal and release transactions.
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
  - Build renewal/release commands, route by backend, map results, and run post-commit effects.
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py`
  - Test renewal/release commands and no-transition mappings.
- Create: `tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py`
  - Direct SQLite renewal/release coverage.
- Create: `tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py`
  - Direct real PostgreSQL renewal/release coverage.
- Modify: `tldw_Server_API/tests/Jobs/parity/scenarios.py`
  - Add release ownership coverage and retain existing renewal parity.
- Modify: `tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py`
  - Run renewal/release scenarios against SQLite.
- Modify: `tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py`
  - Run renewal/release scenarios against real PostgreSQL.
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_lifecycle_side_effects.py`
  - Prove renewal/release events happen once and only after applied transitions.
- Update through Backlog MCP: `TASK-12969` and `TASK-12969.3`.

---

## Task 1: Make Secret Rejection Authoritative

**Files:**
- Create: `tldw_Server_API/tests/Jobs/test_jobs_secret_hygiene.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py:1308-1317`

**Interfaces:**
- Consumes: `JobManager._scan_and_redact_secrets(payload) -> tuple[Any, bool, list[str]]`.
- Produces: unchanged `create_job(...) -> dict[str, Any]`; reject mode raises the existing public `ValueError` before a database connection is opened.

- [x] **Step 1: Add red reject and redact tests**

Create tests with the following behavior:

```python
def test_secret_reject_prevents_insert_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_SECRET_REJECT", "true")
    monkeypatch.delenv("JOBS_SECRET_REDACT", raising=False)
    manager = JobManager(tmp_path / "jobs.db")

    with pytest.raises(ValueError, match="Payload appears to contain secrets"):
        manager.create_job(
            domain="secret-hygiene",
            queue="default",
            job_type="reject",
            payload={"api_key": "do-not-store"},
            owner_user_id="owner-1",
        )

    assert manager.count_jobs(domain="secret-hygiene", owner_user_id="owner-1") == 0


def test_secret_redact_persists_only_redacted_value(tmp_path, monkeypatch):
    monkeypatch.delenv("JOBS_SECRET_REJECT", raising=False)
    monkeypatch.setenv("JOBS_SECRET_REDACT", "true")
    manager = JobManager(tmp_path / "jobs.db")

    created = manager.create_job(
        domain="secret-hygiene",
        queue="default",
        job_type="redact",
        payload={"api_key": "do-not-store"},
        owner_user_id="owner-1",
    )

    assert created["payload"] == {"api_key": "***REDACTED***"}
```

Add a scan-failure compatibility test by monkeypatching `_scan_and_redact_secrets` to raise `RuntimeError("scanner unavailable")`; assert creation still succeeds with the original non-secret payload. This preserves the existing nonfatal scanner-failure contract while distinguishing it from a policy rejection.

Add a real PostgreSQL reject test in the same file:

```python
@pytest.mark.pg_jobs
def test_secret_reject_prevents_insert_postgres(jobs_pg_dsn, monkeypatch):
    monkeypatch.setenv("JOBS_SECRET_REJECT", "true")
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)

    with pytest.raises(ValueError, match="Payload appears to contain secrets"):
        manager.create_job(
            domain="secret-hygiene-pg",
            queue="default",
            job_type="reject",
            payload={"api_key": "do-not-store"},
            owner_user_id="owner-1",
        )

    assert manager.count_jobs(domain="secret-hygiene-pg", owner_user_id="owner-1") == 0
```

- [x] **Step 2: Run the reject test and verify red**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_secret_hygiene.py -q
```

Expected before the fix: `test_secret_reject_prevents_insert_sqlite` fails with `DID NOT RAISE ValueError`.

- [x] **Step 3: Restructure the manager policy block**

Replace the current try/raise/catch block with an `else` branch so only scanner failures are caught:

```python
try:
    cleaned, found, where = self._scan_and_redact_secrets(payload)
except _JOB_NONCRITICAL_EXCEPTIONS as exc:
    logger.debug("Jobs secret hygiene scan error: {}", exc)
else:
    if found and JobManager._is_truthy(os.getenv("JOBS_SECRET_REJECT", "")):
        suffix = "..." if len(where) > 3 else ""
        raise ValueError(f"Payload appears to contain secrets at: {where[:3]}{suffix}")  # noqa: TRY003
    if found:
        payload = cleaned
```

Do not remove `ValueError` from `_JOB_NONCRITICAL_EXCEPTIONS`; that tuple is shared by unrelated post-commit and compatibility paths.

- [x] **Step 4: Verify secret behavior**

Run the new file plus existing JSON and create facade tests:

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_secret_hygiene.py \
  tldw_Server_API/tests/Jobs/test_jobs_json_caps_sqlite.py \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  -q
```

Expected: all selected tests pass.

- [x] **Step 5: Commit Task 1**

```bash
git add tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/tests/Jobs/test_jobs_secret_hygiene.py
git commit -m "fix(jobs): enforce secret rejection before admission"
```

## Task 2: Recover PostgreSQL After Optional Counter Failure

**Files:**
- Create: `tldw_Server_API/tests/Jobs/test_jobs_admission_operations_postgres.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/admission.py:127-142`
- Modify if required: `tldw_Server_API/tests/Jobs/test_jobs_manager_pg_fakes_extended.py`

**Interfaces:**
- Consumes: existing `_bump_counters(cur, command, available_at)`.
- Produces: `_bump_counters_best_effort(...) -> None` with transaction recovery guaranteed before returning.

- [x] **Step 1: Add the real PostgreSQL fault-injection test**

Use the existing `jobs_pg_dsn` fixture and monkeypatch `_bump_counters` to execute invalid SQL:

```python
pytestmark = pytest.mark.pg_jobs


def test_counter_failure_rolls_back_to_savepoint_and_commits_job_event(jobs_pg_dsn, monkeypatch):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")

    def fail_counter(cur, *, command, available_at):
        del command, available_at
        cur.execute("SELECT definitely_missing_column FROM job_counters")

    monkeypatch.setattr(admission, "_bump_counters", fail_counter)
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)

    created = manager.create_job(
        domain="admission-fault",
        queue="default",
        job_type="counter",
        payload={},
        owner_user_id="owner-1",
    )

    assert created["status"] == "queued"
    events = manager.list_job_events_after(after_id=0, domain="admission-fault", limit=10)
    assert [event["event_type"] for event in events] == ["job.created"]
```

- [x] **Step 2: Verify the real PostgreSQL test is red**

Run:

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_admission_operations_postgres.py::test_counter_failure_rolls_back_to_savepoint_and_commits_job_event \
  -q -rs
```

Expected before the fix: `psycopg.errors.InFailedSqlTransaction` at `_insert_created_event`.

- [x] **Step 3: Wrap the optional counter statement in a savepoint**

Implement the helper with a static, non-user-controlled savepoint name:

```python
def _bump_counters_best_effort(
    cur: Any,
    *,
    command: CreateJobCommand,
    available_at: datetime | None,
) -> None:
    cur.execute("SAVEPOINT jobs_admission_counter_update")
    try:
        _bump_counters(cur, command=command, available_at=available_at)
    except _COUNTER_NONCRITICAL_ERRORS as exc:
        cur.execute("ROLLBACK TO SAVEPOINT jobs_admission_counter_update")
        cur.execute("RELEASE SAVEPOINT jobs_admission_counter_update")
        logger.warning(
            "Non-critical Postgres jobs counter update failed for {}:{}:{}: {}",
            command.domain,
            command.queue,
            command.job_type,
            exc,
        )
    else:
        cur.execute("RELEASE SAVEPOINT jobs_admission_counter_update")
```

If savepoint creation, rollback, or release itself fails, let that error propagate. Continuing without restoring transaction validity would repeat the original defect.

- [x] **Step 4: Update strict fake-cursor expectations**

If the existing fake cursor rejects unknown statements, add exact handling for:

```text
SAVEPOINT jobs_admission_counter_update
ROLLBACK TO SAVEPOINT jobs_admission_counter_update
RELEASE SAVEPOINT jobs_admission_counter_update
```

Do not turn the fake into an accept-all cursor; retain assertions for unexpected SQL.

- [x] **Step 5: Verify fault recovery and happy paths**

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_admission_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_manager_pg_fakes_extended.py \
  tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py \
  -q -rs
```

Expected: all selected tests pass with no PostgreSQL skips.

- [x] **Step 6: Commit Task 2**

```bash
git add tldw_Server_API/app/core/Jobs/operations/postgres/admission.py \
  tldw_Server_API/tests/Jobs/test_jobs_admission_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_manager_pg_fakes_extended.py
git commit -m "fix(jobs): isolate optional postgres counter updates"
```

## Task 3: Fail Closed on Quota Query Errors

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/admission.py:145-188`
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/admission.py:149-194`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_admission_operations_postgres.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py`

**Interfaces:**
- Consumes: `_quota_rejection(...) -> AdmissionResult | None`.
- Produces: the same return type for successful queries; database errors propagate and roll back the admission transaction.

- [x] **Step 1: Add focused error-policy tests**

For PostgreSQL, call `_quota_rejection` with a cursor whose `execute` raises `psycopg.ProgrammingError("quota read failed")` and assert that exact error propagates. For SQLite, wrap a connection whose `execute` raises `sqlite3.OperationalError("quota read failed")` for the quota `SELECT` and assert propagation from `create_job_admission`; then assert no job row exists.

The PostgreSQL unit shape is:

```python
class FailingQuotaCursor:
    def execute(self, sql, params=()):
        del sql, params
        raise psycopg.ProgrammingError("quota read failed")


with pytest.raises(psycopg.ProgrammingError, match="quota read failed"):
    admission._quota_rejection(
        FailingQuotaCursor(),
        command=command,
        now=datetime.now(timezone.utc),
        max_queued_quota=1,
        submits_per_minute_quota=0,
    )
```

- [x] **Step 2: Verify red**

Expected before the fix: PostgreSQL `_quota_rejection` returns `None` instead of raising; SQLite continues toward insertion instead of preserving a fail-closed boundary.

- [x] **Step 3: Remove fail-open exception suppression**

Keep the existing quota SQL and rejection messages, but remove the outer `try/except` blocks from both `_quota_rejection` implementations. Preserve these exact branches:

- return `None` immediately when `command.owner_user_id` is absent;
- count `status='queued'` rows for `command.domain` plus `command.owner_user_id` and return `AdmissionResult.rejected(AdmissionRejectionReason.QUOTA_EXCEEDED, message=_MAX_QUEUED_MESSAGE)` when the configured maximum is reached;
- count rows created in the preceding 60 seconds for the same scope and return the same reason with `_SUBMITS_PER_MINUTE_MESSAGE` when that limit is reached;
- return `None` only after every enabled query completes successfully.

Do not convert database failures into `AdmissionResult.rejected(QUOTA_EXCEEDED)`: a backend failure is not evidence that the user exceeded quota. Propagation provides fail-closed behavior without presenting a false policy result.

- [x] **Step 4: Verify quota errors and ordinary quota rejection**

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_admission_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_sqlite.py \
  -q -rs
```

Expected: all selected tests pass and PostgreSQL tests are executed.

- [x] **Step 5: Commit Task 3**

```bash
git add tldw_Server_API/app/core/Jobs/operations/postgres/admission.py \
  tldw_Server_API/app/core/Jobs/operations/sqlite/admission.py \
  tldw_Server_API/tests/Jobs/test_jobs_admission_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py
git commit -m "fix(jobs): fail closed when quota checks fail"
```

## Task 4: Make Enabled Admission Quotas Atomic

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/admission.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/admission.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_quotas_postgres.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_quotas_sqlite.py`

**Interfaces:**
- Consumes: owner/domain-scoped quota values already passed into `create_job_admission`.
- Produces: unchanged public quota `ValueError`; PostgreSQL quota check and insert are serialized by owner/domain, while SQLite uses its shortest available database write transaction only when at least one owner-scoped admission quota is enabled.

- [x] **Step 1: Add deterministic PostgreSQL concurrency coverage**

In the per-test PostgreSQL database, install a temporary trigger that delays `jobs` inserts:

```sql
CREATE FUNCTION jobs_test_delay_insert() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM pg_sleep(0.20);
    RETURN NEW;
END;
$$;

CREATE TRIGGER jobs_test_delay_admission
BEFORE INSERT ON jobs
FOR EACH ROW EXECUTE FUNCTION jobs_test_delay_insert();
```

Construct two managers sequentially, then start two submissions with a `threading.Barrier`. Catch `ValueError` as a quota rejection and assert:

```python
assert sorted(outcome for outcome, _ in results) == ["created", "rejected"]
assert manager.count_jobs(domain="quota-race", owner_user_id="owner-1", status="queued") == 1
```

Use different `job_type` values and no idempotency key so uniqueness does not hide the race. Before the fix, the trigger allows both quota reads to observe zero and both submissions succeed.

- [x] **Step 2: Add deterministic SQLite concurrency coverage**

Create a test-only `JobManager` subclass whose `_connect` registers a `jobs_test_sleep` SQLite function, then install this trigger:

```sql
CREATE TRIGGER jobs_test_delay_admission
BEFORE INSERT ON jobs
BEGIN
    SELECT jobs_test_sleep(0.20);
END;
```

Start two owner/domain-equivalent submissions together. Assert one returns a row, one raises the public quota `ValueError`, neither leaks `sqlite3.OperationalError`, and exactly one queued job remains. Before the fix, both deferred transactions can pass the count and the loser attempts to upgrade a stale read transaction, exposing `database is locked` instead of a quota result.

- [x] **Step 3: Verify both concurrency tests are red**

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_postgres.py::test_max_queued_quota_is_atomic_under_concurrent_admission \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_sqlite.py::test_max_queued_quota_serializes_concurrent_admission \
  -q -rs
```

Expected before the fix: PostgreSQL reports two created rows; SQLite reports either two outcomes inconsistent with the limit or an operational lock error rather than one quota rejection.

- [x] **Step 4: Add a stable PostgreSQL quota lock key**

Add `hashlib` and this helper to `operations/postgres/admission.py`:

```python
def _quota_lock_key(command: CreateJobCommand) -> int:
    material = f"jobs:admission-quota\0{command.domain}\0{command.owner_user_id}".encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=True)
```

At the start of the PostgreSQL transaction, before `_quota_rejection`, acquire the transaction-scoped lock only when needed:

```python
quota_enabled = bool(command.owner_user_id and (max_queued_quota or submits_per_minute_quota))
if quota_enabled:
    cur.execute("SELECT pg_advisory_xact_lock(%s)", (_quota_lock_key(command),))
```

The lock namespace includes a fixed operation prefix so future acquisition locks cannot accidentally share keys merely because owner and domain match.

- [x] **Step 5: Start SQLite quota transactions with `BEGIN IMMEDIATE`**

Before entering the existing `with conn:` transaction:

```python
quota_enabled = bool(command.owner_user_id and (max_queued_quota or submits_per_minute_quota))
if quota_enabled:
    conn.execute("BEGIN IMMEDIATE")

with conn:
    # Existing quota check, insert/idempotency handling, counters, and event write.
```

Do not use `BEGIN IMMEDIATE` when quotas are disabled or there is no owner scope; ordinary admission keeps its current concurrency characteristics.

- [x] **Step 6: Prove lock scope and the SQLite no-quota fast path**

Extend the PostgreSQL concurrency file with submissions for different owner ids under the same domain. Use the delayed insert trigger and overlap events to prove both enter their inserts before either completes, then assert both jobs are created. This protects against replacing the scoped advisory lock with a global PostgreSQL lock.

Add a SQLite direct-operation test with a recording connection proxy. Call `create_job_admission` with `max_queued_quota=0` and `submits_per_minute_quota=0`; assert the recorded statements do not contain `BEGIN IMMEDIATE`. Call again with an owner and `max_queued_quota=1`; assert `BEGIN IMMEDIATE` is the first transaction statement. SQLite cannot keep unrelated quota-enabled writes concurrent because its write lock is database-wide, so the required protection is limiting immediate locking to enabled quota decisions.

- [x] **Step 7: Run the full admission gate**

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_secret_hygiene.py \
  tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_admission_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_precedence_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_precedence_postgres.py \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py \
  tldw_Server_API/tests/Jobs/test_jobs_manager_pg_fakes_extended.py \
  -q -rs
```

Expected: all selected tests pass; no PostgreSQL tests skip.

- [x] **Step 8: Run security and syntax checks**

```bash
python -m compileall -q \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/operations

python -m bandit -r \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/operations/postgres/admission.py \
  tldw_Server_API/app/core/Jobs/operations/sqlite/admission.py \
  -f json -o /tmp/bandit_task_12969_1.json
```

Expected: compile succeeds and Bandit reports no new findings in touched code.

- [x] **Step 9: Review the initial atomicity implementation**

Commit `0cd9fb8f0e` passed strict specification review. Code-quality review identified four hypotheses. Validate them before changing production code:

- idempotent retries can be rejected after their first queued job consumes the quota;
- a PostgreSQL transaction that starts at `REPEATABLE READ` can retain the snapshot established by the advisory-lock query;
- the configured PostgreSQL `lock_timeout` can surface a database error under prolonged contention;
- trigger sleeps widen the race window but do not prove that both pre-fix admissions crossed the quota decision before either insert.

The first, second, and fourth findings are validated for remediation. The third is an intentional consequence of the existing fail-closed database error boundary: keep the bounded database timeout and propagate its error rather than adding implicit whole-transaction retries without a public retry contract.

- [x] **Step 10: Add red review-regression tests**

Add sequential and concurrent idempotent replay tests for both backends with `max_queued=1`; both calls must return the same job and leave one queued row. Add PostgreSQL coverage that begins admissions from `REPEATABLE READ` connections and still admits exactly one same-scope job. Replace timing-only same-scope coordination with test connection/cursor events that force the pre-fix quota-read-to-insert race while allowing the serialized implementation to complete. Ensure every worker thread is joined in cleanup paths.

- [x] **Step 11: Preserve replay semantics and pin the PostgreSQL quota transaction snapshot**

Inside the serialized transaction, resolve an existing idempotency tuple before quota evaluation. Skip quota evaluation only for an existing tuple, then continue through the existing replay path so durable event and facade behavior remain unchanged. For PostgreSQL quota-enabled admissions, set the current transaction to `READ COMMITTED` before the advisory-lock `SELECT`, ensuring the quota query receives a fresh statement snapshot after any wait.

- [x] **Step 12: Re-run review, admission, security, and syntax gates**

Repeat the focused red/green review tests, the Step 7 matrix with required real PostgreSQL execution and no skips, both review stages, compile checks, Ruff on changed files, and Bandit on touched production files.

The first quality re-review validated one additional PostgreSQL race: `prune_jobs` can delete configured nonterminal statuses without acquiring the admission advisory lock. Lock the exact replay row during the pre-quota probe so a concurrent delete cannot turn an observed replay into a fresh quota-bypassed insert. Add a deterministic real-PostgreSQL delete/replay regression, then repeat both review stages and every gate above.

- [x] **Step 13: Commit and open PR 1**

```bash
git add tldw_Server_API/app/core/Jobs/operations/postgres/admission.py \
  tldw_Server_API/app/core/Jobs/operations/sqlite/admission.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_sqlite.py \
  backlog/tasks/task-12969* \
  Docs/superpowers/plans/2026-07-14-jobs-admission-hardening-and-lease-lifecycle.md
git commit -m "fix(jobs): serialize owner scoped admission quotas"
```

Open a PR against `dev` containing only `TASK-12969.1` implementation and planning/tracking updates. Do not include lifecycle contract or operation files. Request code review and address comments using `superpowers:receiving-code-review`.

## Task 5: Enforce the PR Boundary Before Acquisition Work

**Files:**
- Update through Backlog MCP: `TASK-12969.1`, `TASK-12969.2`, `TASK-12969`

**Interfaces:**
- Consumes: merged PR 1 and current `origin/dev`.
- Produces: a new clean acquisition worktree/branch based on the merge commit.

- [ ] **Step 1: Confirm the admission PR is merged and green**

Record the PR URL, merge commit, focused test results, Bandit result, and requester-owned Change summary in `TASK-12969.1`. Mark it Done only after the merge is visible on `origin/dev`.

- [ ] **Step 2: Create a new acquisition worktree**

```bash
git fetch origin dev
git worktree add .worktrees/jobs-lease-acquisition \
  -b codex/jobs-lease-acquisition origin/dev
```

Do not continue acquisition work on the admission branch. Set `TASK-12969.2` to In Progress and record the new worktree/branch.

- [ ] **Step 3: Re-run the merged admission gate**

Run the Task 4 admission gate in the new worktree. Expected: all tests pass with real PostgreSQL execution. Stop and repair regression fallout before any acquisition edits.

## Task 6: Add the Typed Single-Job Acquisition Command

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/operations/contracts.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py`

**Interfaces:**
- Consumes: existing `LifecycleResult`, `NoTransitionReason`, and manager arguments.
- Produces: `AcquireJobCommand` used by both backend modules.

- [ ] **Step 1: Add red command and invariant tests**

Test exact field preservation, frozen behavior, invalid ordering values, and invalid lease duration. Add no-transition coverage for `NO_ELIGIBLE_JOB`.

- [ ] **Step 2: Add the acquisition command dataclass**

Implement these contracts:

```python
@dataclass(frozen=True)
class AcquireJobCommand:
    domain: str
    queue: str
    lease_seconds: int
    worker_id: str
    lease_id: str
    owner_user_id: str | None = None
    job_type: str | None = None
    max_inflight_quota: int = 0
    priority_direction: str = "ASC"
    tie_break: str | None = None
    single_update: bool = False

    def __post_init__(self) -> None:
        if self.priority_direction not in {"ASC", "DESC"}:
            raise ValueError("priority_direction must be ASC or DESC")
        if self.tie_break not in {None, "fifo", "lifo"}:
            raise ValueError("tie_break must be fifo, lifo, or None")
        if self.lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")


```

Extend `NoTransitionReason` with:

```python
NO_ELIGIBLE_JOB = "no_eligible_job"
```

Preserve `tie_break=None`: PostgreSQL currently resolves it to FIFO, while SQLite keeps the current Chatbooks dynamic default and FIFO for other domains.

- [ ] **Step 3: Export and verify contracts**

Add the command classes to `__all__`, run:

```bash
RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py -q
```

Expected: all contract tests pass, including the recursive no-`JobManager` import guard.

- [ ] **Step 4: Commit Task 6**

```bash
git add tldw_Server_API/app/core/Jobs/operations/contracts.py \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py
git commit -m "refactor(jobs): define single job acquisition command"
```

## Task 7: Add the Public Acquisition Parity Safety Net

**Files:**
- Modify: `tldw_Server_API/tests/Jobs/parity/scenarios.py`
- Modify: `tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py`
- Modify: `tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py`

**Interfaces:**
- Consumes: current public `JobManager` methods.
- Produces: passing backend-neutral acquisition scenarios that remain unchanged while Tasks 8-9 move the implementation.

- [ ] **Step 1: Add shared public parity scenarios**

Add these helpers to `parity/scenarios.py` and wrappers in both backend parity files. Add imports for `ThreadPoolExecutor`, `Barrier`, and `Callable`.

```python
LeaseExpiry = Callable[[JobManager, int], None]


def run_acquire_contention_scenario(make_manager: ManagerFactory) -> None:
    seed = make_manager()
    job = seed.create_job(
        domain="parity-contention",
        queue="default",
        job_type="single",
        payload={},
        owner_user_id="owner-1",
    )
    managers = [make_manager(), make_manager()]
    barrier = Barrier(2)

    def acquire(item: tuple[JobManager, str]) -> dict[str, object] | None:
        manager, worker_id = item
        barrier.wait(timeout=10)
        return manager.acquire_next_job(
            domain="parity-contention",
            queue="default",
            lease_seconds=30,
            worker_id=worker_id,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(acquire, zip(managers, ("worker-1", "worker-2"), strict=True)))

    acquired = [result for result in results if result is not None]
    assert len(acquired) == 1
    assert int(acquired[0]["id"]) == int(job["id"])


def run_expired_lease_reclaim_scenario(
    make_manager: ManagerFactory,
    expire_lease: LeaseExpiry,
) -> None:
    manager = make_manager()
    job = manager.create_job(
        domain="parity-expiry",
        queue="default",
        job_type="reclaim",
        payload={},
        owner_user_id="owner-1",
    )
    first = manager.acquire_next_job(
        domain="parity-expiry",
        queue="default",
        lease_seconds=30,
        worker_id="worker-1",
    )
    assert first is not None
    expire_lease(manager, int(job["id"]))

    second = manager.acquire_next_job(
        domain="parity-expiry",
        queue="default",
        lease_seconds=30,
        worker_id="worker-2",
    )
    assert second is not None
    assert int(second["id"]) == int(job["id"])
    assert second["worker_id"] == "worker-2"
    assert second["lease_id"] != first["lease_id"]


```

Construct PostgreSQL managers before entering the executor so concurrent schema initialization cannot interfere with the contention assertion.

- [ ] **Step 2: Add backend expiry adapters in the wrapper files**

The SQLite wrapper passes this callback:

```python
def _expire_sqlite_lease(manager: JobManager, job_id: int) -> None:
    conn = manager._connect()
    try:
        with conn:
            conn.execute(
                "UPDATE jobs SET leased_until = DATETIME('now', '-10 seconds') WHERE id = ?",
                (job_id,),
            )
    finally:
        conn.close()
```

The PostgreSQL wrapper passes this callback:

```python
def _expire_postgres_lease(manager: JobManager, job_id: int) -> None:
    conn = manager._connect()
    try:
        with conn, manager._pg_cursor(conn) as cur:
            cur.execute(
                "UPDATE jobs SET leased_until = NOW() - interval '10 seconds' WHERE id = %s",
                (job_id,),
            )
    finally:
        conn.close()
```

Add one wrapper test per new scenario in each parity file.

- [ ] **Step 3: Verify the characterization suite is green before extraction**

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py \
  -q -rs
```

Expected: every shared scenario passes on both backends and no PostgreSQL test skips. If a scenario fails, correct the scenario or split a newly discovered behavior defect into a separate task before extraction.

- [ ] **Step 4: Commit the green characterization tests**

```bash
git add tldw_Server_API/tests/Jobs/parity
git commit -m "test(jobs): characterize single job acquisition"
```

## Task 8: Implement and Route SQLite Acquisition

**Files:**
- Create: `tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/__init__.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_sqlite.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_lifecycle_side_effects.py`
- Test: `tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py`

**Interfaces:**
- Consumes: the Task 6 acquisition command and a fresh SQLite connection.
- Produces:

The module exposes one function with this exact signature:

- `acquire_job(conn: sqlite3.Connection, *, command: AcquireJobCommand, counters_enabled: bool, now: datetime) -> LifecycleResult`

- [ ] **Step 1: Write and run the red SQLite direct-operation tests**

Import `acquire_job` from the future SQLite lifecycle module. Cover applied acquisition, no eligible row, FIFO/LIFO/default ordering, dependency blocking, expired reclaim, max-inflight quota, and counter movement. Run:

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_sqlite.py \
  -q
```

Expected: collection fails because `operations.sqlite.lifecycle` does not exist.

- [ ] **Step 2: Implement SQLite acquire transaction**

Move the existing eligibility, dependency, ordering, single-update toggle, expired-lease reclaim, update, row fetch, and counter SQL into `acquire_job`. Preserve dynamic Chatbooks ordering exactly. When `max_inflight_quota > 0` and `owner_user_id` exists, execute `BEGIN IMMEDIATE` before count plus acquisition so both are one serialized decision.

Use `cursor.rowcount` for transition success. Return:

```python
LifecycleResult.applied(row=dict(row))
LifecycleResult.no_transition(NoTransitionReason.NO_ELIGIBLE_JOB)
```

Counter updates remain inside the transaction. Do not decrypt payloads, emit events, update gauges, record SLA breaches, or observe metrics in the operation module.

- [ ] **Step 3: Export acquisition and route the SQLite facade**

`JobManager` continues to:
- honor acquire gate and queue pause;
- clamp/adapt lease seconds;
- resolve priority direction, tie-break, single-update flag, and quota values;
- generate the lease id;
- decrypt/parse payload and assert public invariants;
- record SLA breach after operation commit using its own connection;
- emit metrics, gauges, spans, and in-process events after commit;
- map acquisition no-transition to `None`.

The facade call shape is:

```python
result = _sqlite_acquire_job(
    conn,
    command=AcquireJobCommand(
        domain=domain,
        queue=queue,
        lease_seconds=lease_seconds,
        worker_id=worker_id,
        lease_id=str(_uuid.uuid4()),
        owner_user_id=owner_user_id,
        job_type=job_type,
        max_inflight_quota=self._quota_get("JOBS_QUOTA_MAX_INFLIGHT", domain, owner_user_id),
        priority_direction=self._priority_dir_for(domain, backend="sqlite"),
        tie_break=self._tie_break_for(domain, backend="sqlite"),
        single_update=JobManager._is_truthy(os.getenv("JOBS_SQLITE_SINGLE_UPDATE_ACQUIRE", "")),
    ),
    counters_enabled=JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")),
    now=self._clock.now_utc(),
)
```

- [ ] **Step 4: Add and satisfy facade post-commit side-effect tests**

In `test_jobs_lifecycle_side_effects.py`, monkeypatch `emit_job_event`, `observe_queue_latency`, and `_update_gauges`. Stub `_sqlite_acquire_job` with applied and no-transition `LifecycleResult` values. Assert applied acquisition emits one `job.acquired`; no-transition or raised backend errors emit no success event, metric, or gauge. Record `"operation-returned"` in the stub and assert every callback occurs later in the recorded order.

- [ ] **Step 5: Verify SQLite direct and facade behavior**

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_side_effects.py \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  tldw_Server_API/tests/Jobs/test_jobs_manager_acquire.py \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_ordering_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_parallel_acquire_sqlite.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Confirm renewal and release are untouched**

Use `git diff --function-context origin/dev -- tldw_Server_API/app/core/Jobs/manager.py` and verify `renew_job_lease`, `release_job`, `batch_renew_leases`, and terminal methods contain no behavioral edits. Import-only adjacency changes must be reviewed explicitly.

- [ ] **Step 7: Commit Task 8**

```bash
git add tldw_Server_API/app/core/Jobs/operations/sqlite \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_side_effects.py
git commit -m "refactor(jobs): extract sqlite lease acquisition"
```

## Task 9: Implement and Route PostgreSQL Acquisition

**Files:**
- Create: `tldw_Server_API/app/core/Jobs/operations/postgres/lifecycle.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/__init__.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_postgres.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_lifecycle_side_effects.py`
- Test: `tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py`

**Interfaces:**
- Consumes: the Task 6 acquisition command, a psycopg connection, and the existing cursor factory.
- Produces:

The module exposes one function with this exact signature:

- `acquire_job(conn: Any, cursor_factory: Callable[[Any], AbstractContextManager[Any]], *, command: AcquireJobCommand, counters_enabled: bool, now: datetime) -> LifecycleResult`

- [ ] **Step 1: Write and run the red PostgreSQL direct-operation tests**

Import `acquire_job` from the future PostgreSQL lifecycle module. Use `pytestmark = pytest.mark.pg_jobs` and `jobs_pg_dsn`. Cover applied acquisition, no eligible row, `SKIP LOCKED` contention, priority/tie ordering, dependency blocking, expired reclaim, atomic max-inflight quota, counter movement, and counter savepoint recovery. Run:

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_postgres.py \
  -q -rs
```

Expected: collection fails because `operations.postgres.lifecycle` does not exist. The test must execute rather than skip.

- [ ] **Step 2: Implement PostgreSQL acquire transaction**

Move both existing `JOBS_PG_SINGLE_UPDATE_ACQUIRE` and two-step `FOR UPDATE SKIP LOCKED` paths into the operation. Preserve current ordering and dependency predicates. If max-inflight quota is enabled for an owner, acquire a transaction-scoped advisory lock in the namespace `jobs:acquire-inflight`, count active unexpired processing rows, and return `NO_ELIGIBLE_JOB` at the limit before selecting another row.

Generate the advisory key with the same stable BLAKE2b technique as admission but a distinct fixed prefix. Keep counter updates in a savepoint so a noncritical counter failure cannot poison the lease transaction.

- [ ] **Step 3: Route PostgreSQL facade calls and remove migrated acquisition SQL**

Use the same facade responsibilities and result mapping established for SQLite. Delete the old PostgreSQL acquisition SQL only after direct and parity tests pass. Keep renewal, release, batch renewal, and terminal methods byte-for-byte unchanged except import/format adjustments required by tooling.

- [ ] **Step 4: Extend post-commit side-effect tests to PostgreSQL routing**

Add applied, no-transition, and raised-error stubs for `_postgres_acquire_job`. Reuse the Task 8 event-order assertions and verify facade behavior is identical across backend routing.

- [ ] **Step 5: Verify direct PostgreSQL and parity behavior**

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_postgres.py \
  tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_ordering_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_pg_single_update_acquire_toggle.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_postgres.py \
  -q -rs
```

Expected: all selected tests pass and none skip.

- [ ] **Step 6: Run focused concurrency stress**

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_pg_concurrency_stress.py \
  tldw_Server_API/tests/Jobs/test_jobs_pg_single_update_acquire_stress.py \
  -q -rs
```

Expected: both stress files pass. Record runtime and test counts in `TASK-12969.2`.

- [ ] **Step 7: Confirm renewal and release are untouched**

Repeat the Task 8 function-context review for `manager.py`. The acquisition PR must not move, rewrite, or opportunistically clean up renewal/release code.

- [ ] **Step 8: Commit Task 9**

```bash
git add tldw_Server_API/app/core/Jobs/operations/postgres \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_side_effects.py
git commit -m "refactor(jobs): extract postgres lease acquisition"
```

## Task 10: Final Acquisition Verification and PR 2

**Files:**
- Modify: `Docs/superpowers/plans/2026-07-14-jobs-admission-hardening-and-lease-lifecycle.md`
- Update through Backlog MCP: `TASK-12969`, `TASK-12969.2`

**Interfaces:**
- Consumes: completed Tasks 6-9.
- Produces: a review-ready PR against `dev` containing only the single-job acquisition extraction.

- [ ] **Step 1: Run the focused two-backend matrix**

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_side_effects.py \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py \
  tldw_Server_API/tests/Jobs/test_jobs_manager_acquire.py \
  tldw_Server_API/tests/Jobs/test_jobs_parallel_acquire_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_ordering_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_acquire_ordering_postgres.py \
  -q -rs
```

Expected: all selected tests pass and no PostgreSQL test skips.

- [ ] **Step 2: Verify boundaries mechanically**

```bash
rg -n "JobManager" tldw_Server_API/app/core/Jobs/operations
rg -n "def (acquire_next_job|renew_job_lease|release_job)" tldw_Server_API/app/core/Jobs/manager.py
rg -n "UPDATE jobs|SELECT .* FROM jobs" \
  tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py \
  tldw_Server_API/app/core/Jobs/operations/postgres/lifecycle.py
```

Expected:
- no operation module imports or names `JobManager`;
- all three public facade methods still exist;
- only migrated acquisition SQL is present in lifecycle modules;
- renewal, release, `batch_renew_leases`, and terminal SQL remain in manager and are explicitly listed as deferred.

- [ ] **Step 3: Run syntax and security validation**

```bash
python -m compileall -q \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/operations

python -m bandit -r \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/operations \
  -f json -o /tmp/bandit_task_12968_2.json
```

Expected: compile succeeds and Bandit reports no new findings.

- [ ] **Step 4: Review scope and diff**

Confirm the PR does not contain:
- admission behavior changes beyond merged PR 1;
- renewal or release extraction;
- terminal transition extraction;
- batch renewal changes;
- schema or migration changes;
- public API response changes;
- unrelated formatting or generated metadata.

- [ ] **Step 5: Commit tracking updates and open PR 2**

```bash
git add Docs/superpowers/plans/2026-07-14-jobs-admission-hardening-and-lease-lifecycle.md \
  backlog/tasks/task-12968*
git commit -m "docs(jobs): record acquisition extraction verification"
```

Open the acquisition PR against `dev`, request code review, and include a requester-owned Change summary. Mark `TASK-12969.2` Done only after merge. Keep parent `TASK-12969` In Progress for PR 3.

## Task 11: Gate PR 3 and Characterize Renewal/Release

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/operations/contracts.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py`
- Modify: `tldw_Server_API/tests/Jobs/parity/scenarios.py`
- Modify: `tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py`
- Modify: `tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py`
- Update through Backlog MCP: `TASK-12969.2`, `TASK-12969.3`, `TASK-12969`

**Interfaces:**
- Consumes: merged PR 2 and current `origin/dev`.
- Produces: typed `RenewLeaseCommand` and `ReleaseJobCommand` contracts plus a green public parity safety net.

- [ ] **Step 1: Confirm PR 2 is merged and create a fresh worktree**

Record PR 2's URL, merge commit, tests, Bandit result, and requester-owned Change summary in `TASK-12969.2`, then create:

```bash
git fetch origin dev
git worktree add .worktrees/jobs-lease-renew-release \
  -b codex/jobs-lease-renew-release origin/dev
```

Set `TASK-12969.3` to In Progress. Re-run the Task 10 acquisition matrix before editing renewal/release code. A regression blocks PR 3 work.

- [ ] **Step 2: Add red command tests, then implement the contracts**

Test exact field preservation and frozen behavior, then add:

```python
@dataclass(frozen=True)
class RenewLeaseCommand:
    job_id: int
    seconds: int
    worker_id: str | None = None
    lease_id: str | None = None
    progress_percent: float | None = None
    progress_message: str | None = None
    enforce: bool = False


@dataclass(frozen=True)
class ReleaseJobCommand:
    job_id: int
    worker_id: str | None = None
    lease_id: str | None = None
    reason: str | None = None
    enforce: bool = False
```

Use `WRONG_STATUS` for a present non-processing row, `MISSING` for an absent id, and the existing `STALE_LEASE` for enforced worker/token mismatch. Do not create separate worker/token mismatch reason variants unless public behavior requires them.

- [ ] **Step 3: Add the shared public release scenario**

Add this helper to `parity/scenarios.py` and wrappers to both backend parity files:

```python
def run_release_lease_ownership_scenario(make_manager: ManagerFactory) -> None:
    manager = make_manager()
    job = manager.create_job(
        domain="parity-release",
        queue="default",
        job_type="release",
        payload={},
        owner_user_id="owner-1",
    )
    acquired = manager.acquire_next_job(
        domain="parity-release",
        queue="default",
        lease_seconds=30,
        worker_id="worker-1",
    )
    assert acquired is not None
    lease_id = str(acquired["lease_id"])

    assert manager.release_job(
        int(job["id"]), worker_id="worker-2", lease_id=lease_id, enforce=True
    ) is False
    assert manager.release_job(
        int(job["id"]), worker_id="worker-1", lease_id="stale-lease", enforce=True
    ) is False
    current = manager.get_job(int(job["id"]))
    assert current is not None
    assert current["status"] == "processing"

    assert manager.release_job(
        int(job["id"]),
        worker_id="worker-1",
        lease_id=lease_id,
        reason="yield",
        enforce=True,
    ) is True
    released = manager.get_job(int(job["id"]))
    assert released is not None
    assert released["status"] == "queued"
    for field in ("leased_until", "worker_id", "lease_id", "acquired_at", "started_at"):
        assert released.get(field) is None
```

Retain and run the existing stale-renewal parity scenario. Add public success coverage for no-shorten renewal and progress fields if those behaviors are not already asserted by both backend wrappers.

- [ ] **Step 4: Verify and commit only green contracts/characterization**

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_progress_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_progress_postgres.py \
  tldw_Server_API/tests/Jobs/test_fairness_and_renew.py \
  -q -rs
```

Expected: all tests pass and PostgreSQL does not skip. Commit only after the characterization suite is green; do not make a red test-only commit.

## Task 12: Extract SQLite and PostgreSQL Renewal/Release

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/__init__.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/lifecycle.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/__init__.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_lifecycle_side_effects.py`

**Interfaces:**
- Consumes: Task 11 contracts and existing lifecycle modules from merged PR 2.
- Produces:
  - `renew_lease(..., command: RenewLeaseCommand, now: datetime) -> LifecycleResult`
  - `release_job(..., command: ReleaseJobCommand, counters_enabled: bool, now: datetime) -> LifecycleResult`

- [ ] **Step 1: Write direct operation tests, then implement SQLite**

In the same task, first run red direct tests for missing job, wrong status, stale enforced worker/lease identity, no-shorten renewal, progress updates, release field clearing, and counter movement. Then:

- preserve SQLite's maximum-of-current-lease-and-requested-expiry renewal expression;
- build optional progress updates with parameterized SQL;
- classify zero-row results in the same transaction;
- validate release ownership before mutation;
- clear every existing lease/start field on release;
- keep counter updates in the durable transaction;
- return `LifecycleResult` without manager callbacks.

Route the SQLite facade only after direct tests pass. The facade maps no-transition to `False` and retains validation, compatibility behavior, and post-commit effects.

- [ ] **Step 2: Write direct operation tests, then implement PostgreSQL**

Use `pytestmark = pytest.mark.pg_jobs` and `jobs_pg_dsn`; a skip is failure. Preserve:

- `UPDATE ... RETURNING *` for renewal;
- `GREATEST(COALESCE(leased_until, now), now + interval)` no-shorten behavior;
- optional progress fields and enforcement predicates;
- zero-row classification inside the transaction;
- `FOR UPDATE` release validation;
- counter savepoint isolation so optional counter failures cannot poison release commit.

Route the PostgreSQL facade only after direct tests pass. Delete only the migrated single-job renewal/release SQL.

- [ ] **Step 3: Prove post-commit side effects**

Extend `test_jobs_lifecycle_side_effects.py` with applied, no-transition, and raised-error stubs for both backends. Assert:

- applied renewal emits one `job.lease_renewed` after operation return;
- applied release with a reason emits one `job.released` after operation return;
- no-transition and backend errors emit no success event, metric, or gauge;
- acquisition tests from PR 2 remain unchanged and green.

- [ ] **Step 4: Run the focused renewal/release matrix**

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_side_effects.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_progress_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_progress_postgres.py \
  tldw_Server_API/tests/Jobs/test_fairness_and_renew.py \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py \
  -q -rs
```

Expected: all tests pass and PostgreSQL does not skip.

- [ ] **Step 5: Commit backend work in reviewable units**

Commit SQLite routing after its direct/facade matrix passes, then PostgreSQL routing after its real-database matrix passes. Do not commit red tests. Keep both commits in the same PR 3 branch so the final parity review sees one coherent transition family.

## Task 13: Final Renewal/Release Verification and PR 3

**Files:**
- Modify: `Docs/superpowers/plans/2026-07-14-jobs-admission-hardening-and-lease-lifecycle.md`
- Update through Backlog MCP: `TASK-12969`, `TASK-12969.3`

**Interfaces:**
- Consumes: completed Tasks 11-12.
- Produces: a review-ready PR against `dev` containing only single-job renewal/release extraction.

- [ ] **Step 1: Run the two-backend regression matrix**

Run the Task 10 acquisition matrix plus the Task 12 renewal/release matrix, then run these unchanged neighboring paths explicitly:

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_batch_lifecycle_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_batch_lifecycle_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_finalize_idempotency_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_finalize_idempotency_postgres.py \
  tldw_Server_API/tests/Jobs/test_enforcement.py \
  tldw_Server_API/tests/Jobs/test_jobs_status_guardrails.py \
  tldw_Server_API/tests/Jobs/test_jobs_status_guardrails_postgres.py \
  -q -rs
```

Expected: batch and terminal behavior remains green on both backends and PostgreSQL does not skip.

- [ ] **Step 2: Verify boundaries mechanically**

```bash
rg -n "JobManager" tldw_Server_API/app/core/Jobs/operations
rg -n "def (acquire_next_job|renew_job_lease|release_job|batch_renew_leases)" \
  tldw_Server_API/app/core/Jobs/manager.py
git diff --function-context origin/dev -- tldw_Server_API/app/core/Jobs/manager.py
```

Expected: operation modules do not reference `JobManager`; public methods remain; only single-job renew/release SQL moved; acquisition behavior is unchanged from merged PR 2; batch and terminal methods have no behavioral changes.

- [ ] **Step 3: Run syntax and security validation**

```bash
python -m compileall -q \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/operations

python -m bandit -r \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/operations \
  -f json -o /tmp/bandit_task_12968_3.json
```

Expected: compile succeeds and Bandit reports no new findings.

- [ ] **Step 4: Open PR 3 and finalize tracking after merge**

Confirm no schema, batch, terminal, admission, or unrelated formatting changes. Commit plan/Backlog updates, open PR 3 against `dev`, request review, and include a requester-owned Change summary. Mark `TASK-12969.3` and parent `TASK-12969` Done only after the merge is visible on `origin/dev` and all evidence is recorded.

---

## Two-Week Execution Sequence

| Day | Deliverable | Gate |
|---|---|---|
| 1 | Task 1 secret rejection tests and fix | SQLite secret suite green |
| 2 | Task 2 PostgreSQL savepoint recovery | Real PostgreSQL fault injection green |
| 3 | Task 3 fail-closed quota policy | Direct backend error tests green |
| 4 | Task 4 atomic quota implementation | Deterministic SQLite/PostgreSQL concurrency green |
| 5 | Admission full gate, Bandit, review, PR 1 | Merge required before Day 6 code work |
| 6 | Fresh acquisition worktree, contract, and public characterization | Admission gate green on latest dev |
| 7 | SQLite acquisition extraction | SQLite acquisition matrix green; renew/release untouched |
| 8 | PostgreSQL acquisition extraction, Bandit, PR 2 | Real PostgreSQL stress green; merge required before Day 9 |
| 9 | Fresh renewal/release worktree, contracts, characterization, and backend extraction | PR 2 gate green; direct and parity tests green |
| 10 | Renewal/release regression gate, Bandit, PR 3 | No batch/terminal scope leakage; human Change summary present |

The calendar is intentionally contingent on review turnaround. If PR 1 or PR 2 has not merged by the next gate, use the remaining time for review response, added characterization, and plan refinement; do not stack dependent production changes on an unmerged branch merely to preserve the day labels.

The committed two-week outcome is PR 1 merged and PR 2 either merged or review-ready with real PostgreSQL evidence. PR 3 is a stretch outcome: start its production edits only if PR 2 merges early enough to preserve the fresh-base gate. Otherwise, finish Task 11 characterization and leave TASK-12969.3 ready for the next owner without carrying an unreviewed branch stack.

## Rollback Boundaries

- PR 1 is independently revertible because it changes only admission policy/transaction handling and its tests.
- PR 2 is independently revertible because it moves only acquisition while public calls remain behind `JobManager`; renewal/release stay on the prior implementation.
- PR 3 is independently revertible because it moves only renewal/release behind the same facade and contains no data migration.
- Do not combine the PRs. A lifecycle regression must not require reverting the security/quota fixes, and a renewal regression must not require reverting acquisition.
- Advisory locks are transaction scoped, require no cleanup, and leave no persistent schema objects. Test-only delay triggers exist only in disposable per-test databases.

## Deferred Work After This Plan

- `batch_renew_leases` extraction and batch transaction semantics.
- Complete/fail/cancel/retry/quarantine lifecycle operation families.
- Admin endpoint state-changing SQL.
- Runtime adoption of the existing `JobsSettings` scaffold.
- Remaining domain adapter compatibility coverage beyond current representative mappings.
- Broader `JobManager` decomposition after the lease slice is measured and reviewed.
