# Jobs Batch Lease Renewal Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `JobManager.batch_renew_leases` into typed, backend-owned SQLite and PostgreSQL operations while preserving its public integer result, input-order behavior, clock timing, no-op semantics, duration clamping, and whole-batch atomicity.

**Architecture:** `JobManager` remains the compatibility facade: it resolves enforcement, opens the connection, normalizes and clamps every item into an immutable command, selects the backend operation, maps `applied_count` to `int`, and closes nonfatally. Each backend operation owns one atomic scope around the complete ordered tuple, samples the injected clock with the existing backend cadence, executes fixed bound SQL without no-op classification reads, and returns an immutable count result. The normal facade path uses a native transaction; a direct SQLite call inside a caller-owned transaction uses a savepoint and leaves that transaction open. Single and batch renewal share only a pure statement-and-parameter builder; their transaction and result handling remain separate.

**Tech Stack:** Python 3.14, frozen dataclasses, sqlite3, psycopg 3, pytest, Hypothesis, existing Jobs PostgreSQL fixtures, Ruff, compileall, Bandit.

## Global Constraints

- Tracking task: `TASK-13010`.
- Approved design: `Docs/superpowers/specs/2026-08-01-jobs-batch-lease-renewal-extraction-design.md`.
- Implementation base: `origin/dev` at `5605b9d990` after the design branch rebase.
- Preserve `JobManager.batch_renew_leases(items: list[dict[str, Any]], *, enforce: bool | None = None) -> int`.
- Open the backend connection before item normalization to preserve connection-failure precedence.
- Complete normalization of every item before dispatching any backend mutation.
- Malformed-item normalization errors intentionally precede backend clock reads,
  PostgreSQL cursor/RLS setup, and backend dispatch.
- Preserve input order, duplicate update-attempt counting, expected zero-row no-ops, non-shortening leases, and one atomic transaction for the complete batch.
- Direct SQLite calls made inside an active caller transaction must isolate the batch with a savepoint, preserving both the outer transaction and unrelated caller-owned work.
- Preserve PostgreSQL one-clock-sample-per-batch behavior, including one clock call for an empty batch.
- Preserve SQLite one-clock-sample-per-item behavior, including zero clock calls for an empty batch.
- Read and apply `JOBS_LEASE_MAX_SECONDS` once per item in the facade; backend operations do not read settings or reclamp durations.
- Do not add batch limits, sorting, deduplication, set-based bulk SQL, schema changes, migrations, events, counters, metrics, or `JobsSettings` adoption.
- Backend operation modules must not import or reference `JobManager`.
- SQL must use fixed variants and bound parameters. Single and batch paths may share only pure SQL/parameter construction, never execution, cursors, transaction contexts, classification reads, or result mapping.
- Each new test module carries exactly one test-type marker. PostgreSQL modules additionally carry `pytest.mark.pg_jobs` as infrastructure metadata.
- Real PostgreSQL evidence is mandatory. Run it with `TLDW_TEST_POSTGRES_REQUIRED=1`; a skip is a failed gate.
- Run Bandit on every touched production path before completion.
- The pull request cannot merge until the human requester supplies a `Change summary` explaining what changed and why these transaction and compatibility choices were used.
- The original provisional `TASK-12989` collided with two records already on `dev`. The Jobs record was manually renumbered to unique `TASK-13010` after independent review, using the requester's prior approval for narrowly scoped manual repair of this specific task file.

---

## Observed Baseline

The documentation-only branch was tested before planning:

```text
SQLite/contracts/property matrix: 69 passed, 144 warnings
PostgreSQL matrix:                26 skipped because PostgreSQL was unreachable
```

The SQLite result came from the existing batch lifecycle, direct renewal/release, contract, and operation-contract property suites. The PostgreSQL skip is an environment limitation, not acceptable final evidence; all PostgreSQL commands below set `TLDW_TEST_POSTGRES_REQUIRED=1` so that the same condition fails visibly during implementation.

## Stage Map

### Stage 1: Public Characterization
**Goal:** Lock current facade behavior before moving SQL.
**Success Criteria:** Focused SQLite and required PostgreSQL tests pass against the inline implementation, including rollback and clock cadence.
**Tests:** New public batch-renew characterization modules.
**Status:** Not Started

### Stage 2: Typed Contracts
**Goal:** Add immutable batch item, command, and result contracts with enforced invariants.
**Success Criteria:** Contract and narrow property tests pass; operation modules remain independent of `JobManager`.
**Tests:** Unit contract tests and Hypothesis result-invariant test.
**Status:** Not Started

### Stage 3: Backend Operations
**Goal:** Add atomic SQLite and PostgreSQL batch operations behind pure SQL builders.
**Success Criteria:** Direct operation tests prove exact counts, no-ops, clock timing, non-shortening, and rollback.
**Tests:** Existing direct renewal/release modules extended per backend.
**Status:** Not Started

### Stage 4: Facade Routing
**Goal:** Replace inline SQL with normalized typed dispatch while retaining the public contract.
**Success Criteria:** Routing tests and all public characterization tests pass unchanged.
**Tests:** Focused unit routing module plus both public characterization modules.
**Status:** Not Started

### Stage 5: Verification And Review
**Goal:** Prove compatibility, security hygiene, and branch readiness.
**Success Criteria:** Required PostgreSQL runs with zero skips; focused and neighboring Jobs suites, Ruff, compileall, Bandit, diff checks, and independent review pass.
**Tests:** Full matrix listed in Task 6.
**Status:** Not Started

## File Structure

- Create: `tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_sqlite.py`
  - Public SQLite compatibility and durable rollback coverage against `JobManager`.
- Create: `tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_postgres.py`
  - Equivalent required real-PostgreSQL public coverage with unique trigger cleanup.
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py`
  - Unit coverage for immutable batch contracts and operation import boundaries.
- Modify: `tldw_Server_API/tests/Jobs/property/test_operation_contract_properties.py`
  - Narrow generated coverage for requested/applied count invariants.
- Modify: `tldw_Server_API/app/core/Jobs/operations/contracts.py`
  - Add `BatchRenewLeaseItem`, `BatchRenewLeasesCommand`, and `BatchRenewLeasesResult`.
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py`
  - Direct SQLite batch operation behavior, clock, and rollback coverage.
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py`
  - Extract the pure renewal statement builder and add atomic `renew_leases_batch`.
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/__init__.py`
  - Export SQLite `renew_leases_batch`.
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py`
  - Direct real-PostgreSQL batch operation behavior, clock, and trigger rollback coverage.
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/lifecycle.py`
  - Extract the pure renewal statement builder and add atomic `renew_leases_batch`.
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/__init__.py`
  - Export PostgreSQL `renew_leases_batch`.
- Create: `tldw_Server_API/tests/Jobs/test_jobs_batch_renew_routing.py`
  - Unit coverage for normalization, connection precedence, typed dispatch, and integer mapping.
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
  - Replace inline SQL with connection-first normalization and backend dispatch.
- Modify through Backlog tooling: `TASK-13010`
  - Record the approved plan, changed files, verification evidence, pull request, and final summary.

---

### Task 1: Characterize The Existing Public Batch Contract

**Files:**
- Create: `tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_sqlite.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_postgres.py`

**Interfaces:**
- Consumes: existing `JobManager.batch_renew_leases(items, *, enforce=None) -> int`.
- Produces: green compatibility tests that remain unchanged when Task 5 routes through backend operations.

- [ ] **Step 1: Add deterministic public test helpers**

Both modules must define a fixed clock and backend-appropriate helpers that insert processing and queued rows, fetch leases through a fresh connection, and return worker/lease identities. Use this clock shape:

```python
class RecordingClock:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.calls = 0

    def now_utc(self) -> datetime:
        self.calls += 1
        return self.now
```

Use `NOW = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)`. SQLite creates a database with `ensure_jobs_tables`, sets `row_factory = sqlite3.Row`, and constructs `JobManager(db_path, clock=clock)`. PostgreSQL uses `jobs_pg_dsn`, disables counters/outbox, and constructs `JobManager(None, backend="postgres", db_url=jobs_pg_dsn, clock=clock)`.

Declare markers exactly as follows:

```python
# SQLite module
pytestmark = pytest.mark.integration

# PostgreSQL module
pytestmark = [pytest.mark.integration, pytest.mark.pg_jobs]
```

- [ ] **Step 2: Characterize counts, no-ops, duplicates, and non-shortening**

Add `test_batch_renew_counts_ordered_attempts_and_preserves_longer_lease_sqlite`
and `test_batch_renew_counts_ordered_attempts_and_preserves_longer_lease_postgres`.
Seed:

```python
valid_id = _insert_job(status="processing", worker_id="worker-1", lease_id="lease-1")
queued_id = _insert_job(status="queued", worker_id=None, lease_id=None)
stale_id = _insert_job(status="processing", worker_id="worker-1", lease_id="lease-1")
long_id = _insert_job(
    status="processing",
    worker_id="worker-1",
    lease_id="lease-1",
    leased_until=NOW + timedelta(minutes=10),
)

items = [
    {"job_id": valid_id, "seconds": 30, "worker_id": "worker-1", "lease_id": "lease-1"},
    {"job_id": 999_999, "seconds": 30, "worker_id": "worker-1", "lease_id": "lease-1"},
    {"job_id": queued_id, "seconds": 30, "worker_id": "worker-1", "lease_id": "lease-1"},
    {"job_id": stale_id, "seconds": 30, "worker_id": "worker-2", "lease_id": "lease-1"},
    {"job_id": valid_id, "seconds": 30, "worker_id": "worker-1", "lease_id": "lease-1"},
    {"job_id": long_id, "seconds": 30, "worker_id": "worker-1", "lease_id": "lease-1"},
]
```

Set enforcement explicitly with `enforce=True`. Assert the returned count is `3`: two attempts against `valid_id` plus one against `long_id`. Assert the queued and stale rows are unchanged and `long_id` remains at `NOW + 10 minutes`.

- [ ] **Step 3: Characterize per-item clamping and backend clock cadence**

Set `JOBS_LEASE_MAX_SECONDS=60`, seed three expired processing jobs, and submit durations `0`, `30`, and `120`. Assert exact count `3` and durable leases at `NOW + 1 second`, `NOW + 30 seconds`, and `NOW + 60 seconds`. Assert:

```python
# PostgreSQL
assert clock.calls == 1

# SQLite
assert clock.calls == 3
```

Add an empty-batch test per backend. Assert result `0`; PostgreSQL clock calls equal `1`, SQLite clock calls equal `0`, and a monkeypatched `os.getenv` guard for `JOBS_LEASE_MAX_SECONDS` is never reached.

- [ ] **Step 4: Characterize malformed-item rollback**

Seed two processing jobs and call with a valid first item followed by `{"job_id": "not-an-int", "seconds": 30}`. Assert `ValueError` and query through a fresh connection to prove the first lease did not change. This test must pass against the inline implementation before extraction.

- [ ] **Step 5: Characterize database-triggered rollback**

For SQLite, assign the second job a unique constant worker identity and create a fixed `BEFORE UPDATE ON jobs` trigger scoped to that identity that executes `RAISE(ABORT, 'forced batch renewal failure')`. For PostgreSQL, generate unique names with `uuid.uuid4().hex`, create a function that raises `forced batch renewal failure`, and compose all dynamic identifiers and the scoped job literal with `psycopg.sql`:

```python
cur.execute(
    psycopg_sql.SQL(
        "CREATE TRIGGER {} BEFORE UPDATE ON jobs "
        "FOR EACH ROW WHEN (OLD.id = {}) EXECUTE FUNCTION {}()"
    ).format(
        psycopg_sql.Identifier(trigger_name),
        psycopg_sql.Literal(second_job_id),
        psycopg_sql.Identifier(function_name),
    )
)
```

Call the public method with the first and second jobs in that order. Assert `sqlite3.IntegrityError` or `psycopg.Error`, then verify both original leases through a fresh connection. PostgreSQL cleanup must run in `finally` through another fresh connection:

```python
with cleanup_connection, manager._pg_cursor(cleanup_connection) as cur:
    cur.execute(
        psycopg_sql.SQL("DROP TRIGGER IF EXISTS {} ON jobs").format(
            psycopg_sql.Identifier(trigger_name)
        )
    )
    cur.execute(
        psycopg_sql.SQL("DROP FUNCTION IF EXISTS {}()").format(
            psycopg_sql.Identifier(function_name)
        )
    )
```

The generated names contain only a fixed ASCII prefix plus `uuid4().hex`; do not interpolate caller-controlled identifiers.

- [ ] **Step 6: Run characterization against the inline implementation**

Run SQLite:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_sqlite.py -q
```

Expected: all tests pass before production routing changes.

Run required PostgreSQL:

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_postgres.py -q
```

Expected: all tests pass with zero skips. If PostgreSQL is unreachable, stop and repair the test environment; do not proceed with skipped evidence.

- [ ] **Step 7: Commit the compatibility safety net**

```bash
git add tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_postgres.py
git commit -m "test(jobs): characterize batch lease renewal"
```

---

### Task 2: Add Immutable Batch Renewal Contracts

**Files:**
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py`
- Modify: `tldw_Server_API/tests/Jobs/property/test_operation_contract_properties.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/contracts.py:91-108,273-283`

**Interfaces:**
- Produces: `BatchRenewLeaseItem`, `BatchRenewLeasesCommand`, and `BatchRenewLeasesResult` exactly as defined below.
- Consumed by: Tasks 3-5 backend operations and facade routing.

- [ ] **Step 1: Write failing unit contract tests**

Import the three new types and add tests with these exact behaviors:

```python
def test_batch_renew_command_snapshots_items_and_is_frozen() -> None:
    source = [BatchRenewLeaseItem(job_id=1, seconds=30)]
    command = BatchRenewLeasesCommand(items=source, enforce=True)  # type: ignore[arg-type]
    source.append(BatchRenewLeaseItem(job_id=2, seconds=45))

    assert command.items == (BatchRenewLeaseItem(job_id=1, seconds=30),)
    with pytest.raises(FrozenInstanceError):
        command.enforce = False


@pytest.mark.parametrize("seconds", [0, -1])
def test_batch_renew_item_rejects_non_positive_normalized_duration(seconds: int) -> None:
    with pytest.raises(ValueError, match="seconds must be positive"):
        BatchRenewLeaseItem(job_id=1, seconds=seconds)


@pytest.mark.parametrize(
    ("requested", "applied"),
    [(-1, 0), (0, -1), (1, 2)],
)
def test_batch_renew_result_rejects_invalid_counts(requested: int, applied: int) -> None:
    with pytest.raises(ValueError):
        BatchRenewLeasesResult(requested_count=requested, applied_count=applied)
```

Also assert an item is frozen, command item order is retained, `BatchRenewLeasesResult(3, 0)` and `(3, 3)` construct successfully, and all three names appear in `contracts.__all__`. Keep the existing AST import-boundary test unchanged; it will cover new operation code later.

- [ ] **Step 2: Write the narrow failing property test**

Extend `test_operation_contract_properties.py`:

```python
@_COMMON
@given(
    requested=st.integers(min_value=-5, max_value=100),
    applied=st.integers(min_value=-5, max_value=105),
)
def test_batch_renew_result_constructs_only_for_valid_count_pairs(
    requested: int,
    applied: int,
) -> None:
    valid = requested >= 0 and 0 <= applied <= requested
    if valid:
        result = BatchRenewLeasesResult(requested_count=requested, applied_count=applied)
        assert result.requested_count == requested
        assert result.applied_count == applied
    else:
        with pytest.raises(ValueError):
            BatchRenewLeasesResult(requested_count=requested, applied_count=applied)
```

- [ ] **Step 3: Run tests and verify red**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py tldw_Server_API/tests/Jobs/property/test_operation_contract_properties.py -q
```

Expected: collection fails because the three batch contract names do not exist.

- [ ] **Step 4: Implement the minimal frozen contracts**

Add after `RenewLeaseCommand`:

```python
@dataclass(frozen=True)
class BatchRenewLeaseItem:
    """One facade-normalized lease renewal attempt."""

    job_id: int
    seconds: int
    worker_id: str | None = None
    lease_id: str | None = None

    def __post_init__(self) -> None:
        if self.seconds < 1:
            raise ValueError("seconds must be positive")


@dataclass(frozen=True)
class BatchRenewLeasesCommand:
    """Ordered immutable lease renewal attempts for one transaction."""

    items: tuple[BatchRenewLeaseItem, ...]
    enforce: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))


@dataclass(frozen=True)
class BatchRenewLeasesResult:
    """Counts produced by one atomic batch renewal operation."""

    requested_count: int
    applied_count: int

    def __post_init__(self) -> None:
        if self.requested_count < 0:
            raise ValueError("requested_count must be non-negative")
        if not 0 <= self.applied_count <= self.requested_count:
            raise ValueError("applied_count must be between zero and requested_count")
```

Export all three through `__all__`. Do not clamp durations or convert job IDs in these contracts; facade normalization owns those policies.

- [ ] **Step 5: Verify green and commit**

```bash
RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py tldw_Server_API/tests/Jobs/property/test_operation_contract_properties.py -q
git add tldw_Server_API/app/core/Jobs/operations/contracts.py tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py tldw_Server_API/tests/Jobs/property/test_operation_contract_properties.py
git commit -m "feat(jobs): add batch lease renewal contracts"
```

---

### Task 3: Implement The Atomic SQLite Batch Operation

**Files:**
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py:36-81,218-259`
- Modify: `tldw_Server_API/app/core/Jobs/operations/sqlite/__init__.py:3-6`

**Interfaces:**
- Consumes: `BatchRenewLeasesCommand` and `Callable[[], datetime]`.
- Produces: `renew_leases_batch(conn, *, command, clock) -> BatchRenewLeasesResult`.
- Preserves: `renew_lease(conn, *, command, now) -> LifecycleResult`.

- [ ] **Step 1: Write failing direct SQLite batch tests**

Import `BatchRenewLeaseItem`, `BatchRenewLeasesCommand`, and `renew_leases_batch`. Extend `_insert_job` to accept unique UUIDs so several rows can coexist. Add direct tests for:

Change the SQLite fixtures so rollback tests can reopen the same database:

```python
@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return ensure_jobs_tables(tmp_path / "jobs.db")


@pytest.fixture()
def conn(db_path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()
```

Rollback tests open `sqlite3.connect(db_path)` after the raised exception, set `row_factory`, query the persisted leases, and close that fresh connection in `finally`.

```python
def test_sqlite_batch_renew_counts_attempts_and_commits_expected_noops(conn):
    valid_id = _insert_job(conn, uuid="valid")
    queued_id = _insert_job(conn, uuid="queued", status="queued")
    stale_id = _insert_job(conn, uuid="stale")
    long_id = _insert_job(conn, uuid="long", leased_until="2026-01-02 13:00:00")
    command = BatchRenewLeasesCommand(
        items=(
            BatchRenewLeaseItem(valid_id, 30, "worker-1", "lease-1"),
            BatchRenewLeaseItem(999_999, 30, "worker-1", "lease-1"),
            BatchRenewLeaseItem(queued_id, 30, "worker-1", "lease-1"),
            BatchRenewLeaseItem(stale_id, 30, "worker-2", "lease-1"),
            BatchRenewLeaseItem(valid_id, 30, "worker-1", "lease-1"),
            BatchRenewLeaseItem(long_id, 30, "worker-1", "lease-1"),
        ),
        enforce=True,
    )

    result = renew_leases_batch(conn, command=command, clock=lambda: NOW)

    assert result == BatchRenewLeasesResult(requested_count=6, applied_count=3)
    assert conn.execute("SELECT leased_until FROM jobs WHERE id = ?", (long_id,)).fetchone()[0] == "2026-01-02 13:00:00"
```

Add a `RecordingClock` assertion that calls equal item count, including no-op items; an empty command returns `(0, 0)` with zero calls. Add `FailOnSecondClock` that returns `NOW` once and then raises `RuntimeError("forced clock failure")`; query from a fresh connection and prove the first update rolled back. Add direct-call coverage proving success and ordinary statement failure leave a caller-owned transaction open, with failure rolling back only the batch savepoint while preserving unrelated caller-owned work. Add a `RAISE(ROLLBACK, ...)` trigger case proving a whole-transaction abort keeps the original `IntegrityError` primary, chains the missing-savepoint cleanup failure, and necessarily closes the caller transaction.

Add a trigger test scoped to the second job using `RAISE(ABORT, 'forced batch renewal failure')`. After the exception, query from a fresh connection and assert both leases are unchanged.

- [ ] **Step 2: Run the direct tests and verify red**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py -q
```

Expected: collection fails because `renew_leases_batch` is not exported.

- [ ] **Step 3: Extract a pure SQLite renewal statement builder**

Move only SQL selection and parameter construction from `renew_lease` into:

```python
def _renew_lease_statement(
    command: RenewLeaseCommand,
    *,
    now: datetime,
) -> tuple[str, tuple[Any, ...]]:
    now_sql = _sqlite_timestamp(now)
    has_percent = command.progress_percent is not None
    has_message = command.progress_message is not None
    sql = _RENEW_SQL_VARIANTS[(command.enforce, has_percent, has_message)]
    params: list[Any] = [now_sql, now_sql, command.seconds]
    if has_percent:
        params.append(float(command.progress_percent))
    if has_message:
        params.append(str(command.progress_message))
    params.append(command.job_id)
    if command.enforce:
        params.extend((command.worker_id, command.lease_id))
    return sql, tuple(params)
```

Change existing `renew_lease` to call this helper, leaving its transaction, classification query, selected row, and `LifecycleResult` mapping unchanged.

- [ ] **Step 4: Implement SQLite `renew_leases_batch`**

```python
def renew_leases_batch(
    conn: sqlite3.Connection,
    *,
    command: BatchRenewLeasesCommand,
    clock: Callable[[], datetime],
) -> BatchRenewLeasesResult:
    """Renew an ordered SQLite lease batch in one atomic scope."""

    applied_count = 0
    with _batch_renew_transaction(conn):
        for item in command.items:
            item_command = RenewLeaseCommand(
                job_id=item.job_id,
                seconds=item.seconds,
                enforce=command.enforce,
                worker_id=item.worker_id,
                lease_id=item.lease_id,
            )
            sql, params = _renew_lease_statement(item_command, now=clock())
            changed = conn.execute(sql, params)
            applied_count += int(changed.rowcount or 0)
        return BatchRenewLeasesResult(
            requested_count=len(command.items),
            applied_count=applied_count,
        )
```

Implement `_batch_renew_transaction` so a fresh connection uses the native connection context, while an already-active connection uses a fixed-name SQLite savepoint that is released on success and rolled back on failure. If SQLite aborts the complete transaction and removes the savepoint, re-raise the primary database error with the cleanup error chained. Import `Callable` and the three batch contracts. Export `renew_leases_batch` from `operations/sqlite/__init__.py`. Do not call `_classify_lifecycle_no_transition` in the batch path.

- [ ] **Step 5: Verify SQLite green and single-renewal compatibility**

```bash
RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_sqlite.py -q
```

Expected: direct tests pass; public tests still exercise the inline manager and remain green.

- [ ] **Step 6: Commit the SQLite operation**

```bash
git add tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py tldw_Server_API/app/core/Jobs/operations/sqlite/__init__.py tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py
git commit -m "refactor(jobs): add atomic SQLite batch renewal"
```

---

### Task 4: Implement The Atomic PostgreSQL Batch Operation

**Files:**
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py`
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/lifecycle.py:56-109,192-225`
- Modify: `tldw_Server_API/app/core/Jobs/operations/postgres/__init__.py:3-6`

**Interfaces:**
- Consumes: `BatchRenewLeasesCommand`, the established cursor factory, and `Callable[[], datetime]`.
- Produces: `renew_leases_batch(conn, cursor_factory, *, command, clock) -> BatchRenewLeasesResult`.
- Preserves: `renew_lease(conn, cursor_factory, *, command, now) -> LifecycleResult`.

- [ ] **Step 1: Write failing direct PostgreSQL batch tests**

Use the existing real `jobs_pg_dsn`, `manager`, `_insert_job`, `_execute`, and
`_fetch_job` helpers. Extend `_insert_job` only if needed to seed distinct status,
identity, and lease values. Construct the direct command explicitly:

```python
valid_id = _insert_job(manager)
queued_id = _insert_job(manager, status="queued", worker_id=None, lease_id=None)
stale_id = _insert_job(manager)
long_expiry = NOW + timedelta(hours=1)
long_id = _insert_job(manager, leased_until=long_expiry)
command = BatchRenewLeasesCommand(
    items=(
        BatchRenewLeaseItem(valid_id, 30, "worker-1", "lease-1"),
        BatchRenewLeaseItem(999_999, 30, "worker-1", "lease-1"),
        BatchRenewLeaseItem(queued_id, 30, "worker-1", "lease-1"),
        BatchRenewLeaseItem(stale_id, 30, "worker-2", "lease-1"),
        BatchRenewLeaseItem(valid_id, 30, "worker-1", "lease-1"),
        BatchRenewLeaseItem(long_id, 30, "worker-1", "lease-1"),
    ),
    enforce=True,
)
clock = RecordingClock(NOW)

result = renew_leases_batch(
    conn,
    manager._pg_cursor,
    command=command,
    clock=clock,
)
```

Assert exact attempt counts, expected no-ops, duplicates, and the longer lease:

```python
assert result == BatchRenewLeasesResult(requested_count=6, applied_count=3)
assert clock.calls == 1
assert _fetch_job(manager, long_id)["leased_until"] == long_expiry
```

Add a second test using `BatchRenewLeasesCommand(items=(), enforce=False)`;
assert `(requested_count, applied_count) == (0, 0)` and one clock call. Wrap
`manager._pg_cursor` in a context manager that increments `cursor_entries` and
delegates to the real factory; assert one entry so the operation cannot bypass
the RLS/session cursor setup.

Add a later-item trigger failure using unique function/trigger names and a fresh cleanup connection in `finally`. Verify both earlier and failing-row leases through `_fetch_job` after rollback. Do not add a fake-cursor transaction test; real PostgreSQL is the required evidence.

- [ ] **Step 2: Run the direct PostgreSQL tests and verify red**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
TLDW_TEST_POSTGRES_REQUIRED=1 RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py -q
```

Expected: collection fails because `renew_leases_batch` is not exported. PostgreSQL unavailability is an environment failure, not the expected red state.

- [ ] **Step 3: Extract a pure PostgreSQL renewal statement builder**

Add this builder using PostgreSQL `_RENEW_SQL_VARIANTS` and native aware
datetimes:

```python
def _renew_lease_statement(
    command: RenewLeaseCommand,
    *,
    now: datetime,
) -> tuple[str, tuple[Any, ...]]:
    has_percent = command.progress_percent is not None
    has_message = command.progress_message is not None
    sql = _RENEW_SQL_VARIANTS[(command.enforce, has_percent, has_message)]
    params: list[Any] = [now, now, command.seconds]
    if has_percent:
        params.append(float(command.progress_percent))
    if has_message:
        params.append(str(command.progress_message))
    params.append(command.job_id)
    if command.enforce:
        params.extend((command.worker_id, command.lease_id))
    return sql, tuple(params)
```

Change existing `renew_lease` to use the builder without changing its
transaction, `fetchone`, no-transition classification, or `LifecycleResult`
mapping.

- [ ] **Step 4: Implement PostgreSQL `renew_leases_batch`**

```python
def renew_leases_batch(
    conn: Any,
    cursor_factory: Callable[[Any], AbstractContextManager[Any]],
    *,
    command: BatchRenewLeasesCommand,
    clock: Callable[[], datetime],
) -> BatchRenewLeasesResult:
    """Renew an ordered PostgreSQL lease batch in one transaction."""

    applied_count = 0
    with conn:
        with cursor_factory(conn) as cur:
            now = clock()
            for item in command.items:
                item_command = RenewLeaseCommand(
                    job_id=item.job_id,
                    seconds=item.seconds,
                    enforce=command.enforce,
                    worker_id=item.worker_id,
                    lease_id=item.lease_id,
                )
                sql, params = _renew_lease_statement(item_command, now=now)
                cur.execute(sql, params)
                if cur.fetchone() is not None:
                    applied_count += 1
            return BatchRenewLeasesResult(
                requested_count=len(command.items),
                applied_count=applied_count,
            )
```

Consuming the existing `RETURNING` row keeps cursor state clean while retaining separate batch result handling. Do not run `_classify_lifecycle_no_transition`; zero returned rows are expected no-ops. Export the function from `operations/postgres/__init__.py`.

- [ ] **Step 5: Verify PostgreSQL green and single-renewal compatibility**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_postgres.py -q
```

Expected: all tests pass with zero skips; public characterization still uses the inline manager at this stage.

- [ ] **Step 6: Commit the PostgreSQL operation**

```bash
git add tldw_Server_API/app/core/Jobs/operations/postgres/lifecycle.py tldw_Server_API/app/core/Jobs/operations/postgres/__init__.py tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py
git commit -m "refactor(jobs): add atomic PostgreSQL batch renewal"
```

---

### Task 5: Route `JobManager.batch_renew_leases`

**Files:**
- Create: `tldw_Server_API/tests/Jobs/test_jobs_batch_renew_routing.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py:62-78,4590-4670`

**Interfaces:**
- Consumes: both backend `renew_leases_batch` functions and all three batch contracts.
- Produces: unchanged public `batch_renew_leases(...) -> int` with connection-first normalization and typed dispatch.

- [ ] **Step 1: Write failing routing tests**

Use `pytestmark = pytest.mark.unit`. Build a minimal manager with `object.__new__(JobManager)` and a fake connection whose `close()` records closure. Monkeypatch the module-level backend function, not transaction internals:

```python
class FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FixedClock:
    def now_utc(self) -> datetime:
        return NOW


def _minimal_manager(backend: str) -> tuple[JobManager, FakeConnection]:
    manager = object.__new__(JobManager)
    connection = FakeConnection()
    manager.backend = backend
    manager._clock = FixedClock()
    manager._connect = lambda: connection
    manager._pg_cursor = lambda conn: nullcontext(conn)
    manager._should_enforce_ack = lambda: True
    return manager, connection
```

Add a parametrized SQLite/PostgreSQL test that supplies ordered seconds `0`, `30`, and `120`, and captures the dispatched command. Wrap `manager_module.os.getenv` so it delegates all other names to the real function, returns `"60"` for `JOBS_LEASE_MAX_SECONDS`, and increments `lease_max_reads`. The backend stub returns:

```python
BatchRenewLeasesResult(requested_count=3, applied_count=2)
```

Assert the facade returns `2`, the command contains job IDs in input order, normalized seconds are `(1, 30, 60)`, `lease_max_reads == 3`, resolved enforcement appears once on the command, worker/lease values are preserved, the correct backend function was called, and the connection closed.

Add these separate behavior tests:

```python
def test_batch_renew_opens_connection_before_normalizing_invalid_input(monkeypatch):
    manager, _ = _minimal_manager("sqlite")
    monkeypatch.setattr(manager, "_connect", lambda: (_ for _ in ()).throw(ConnectionError("offline")))

    with pytest.raises(ConnectionError, match="offline"):
        manager.batch_renew_leases([{"job_id": "invalid", "seconds": 30}], enforce=False)


def test_batch_renew_normalizes_before_clock_cursor_or_backend_dispatch(monkeypatch):
    manager, connection = _minimal_manager("sqlite")
    manager._clock = ExplodingClock()
    called = False

    def backend(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("backend must not run")

    monkeypatch.setattr(manager_module, "_sqlite_renew_leases_batch", backend)

    with pytest.raises(ValueError):
        manager.batch_renew_leases(
            [
                {"job_id": 1, "seconds": 30},
                {"job_id": "invalid", "seconds": 30},
            ],
            enforce=False,
        )

    assert called is False
    assert manager._clock.calls == 0
    assert connection.closed is True
```

Add an empty-command routing test to prove the backend still receives `items=()` instead of the facade returning early. Do not assert cursor contexts, transaction wrappers, function signatures through introspection, or event/metric calls.

- [ ] **Step 2: Run routing tests and verify red**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_batch_renew_routing.py -q
```

Expected: tests fail because the manager still executes inline SQL and does not dispatch typed commands.

- [ ] **Step 3: Add imports and replace the inline implementation**

Import the batch contracts and backend functions using aliases `_sqlite_renew_leases_batch` and `_postgres_renew_leases_batch`. Replace only the body of `batch_renew_leases`:

```python
def batch_renew_leases(
    self,
    items: list[dict[str, Any]],
    *,
    enforce: bool | None = None,
) -> int:
    if enforce is None:
        enforce = self._should_enforce_ack()
    conn = self._connect()
    try:
        command = BatchRenewLeasesCommand(
            items=tuple(
                BatchRenewLeaseItem(
                    seconds=max(
                        1,
                        min(
                            int(os.getenv("JOBS_LEASE_MAX_SECONDS", "3600") or "3600"),
                            int(item.get("seconds") or 0),
                        ),
                    ),
                    job_id=int(item.get("job_id")),
                    worker_id=item.get("worker_id"),
                    lease_id=item.get("lease_id"),
                )
                for item in items
            ),
            enforce=bool(enforce),
        )
        if self.backend == "postgres":
            result = _postgres_renew_leases_batch(
                conn,
                self._pg_cursor,
                command=command,
                clock=self._clock.now_utc,
            )
        else:
            result = _sqlite_renew_leases_batch(
                conn,
                command=command,
                clock=self._clock.now_utc,
            )
        return int(result.applied_count)
    finally:
        _close_connection_nonfatal(conn, operation="batch lease renewal")
```

Preserve the existing public signature on one line if project formatting keeps it that way. Do not add facade events or early-return the empty command.

- [ ] **Step 4: Verify routing and unchanged public behavior**

```bash
RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_batch_renew_routing.py tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py -q
TLDW_TEST_POSTGRES_REQUIRED=1 RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_postgres.py tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py -q
```

Expected: all tests pass, with zero PostgreSQL skips.

- [ ] **Step 5: Commit facade routing**

```bash
git add tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/tests/Jobs/test_jobs_batch_renew_routing.py
git commit -m "refactor(jobs): route batch renewal through operations"
```

---

### Task 6: Run Final Gates And Prepare Review

**Files:**
- Modify through Backlog tooling: `TASK-13010`
- Review all files listed in the File Structure section.

**Interfaces:**
- Consumes: the complete branch implementation.
- Produces: verified, review-ready commits without changing public behavior.

- [ ] **Step 1: Run the complete focused SQLite/contracts matrix**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_batch_renew_routing.py \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py \
  tldw_Server_API/tests/Jobs/property/test_operation_contract_properties.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_batch_lifecycle_sqlite.py \
  -q
```

Expected: all pass.

- [ ] **Step 2: Run required real PostgreSQL coverage**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_batch_lifecycle_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_rls_postgres.py \
  -q
```

Expected: all pass and the pytest summary contains zero skips.

- [ ] **Step 3: Run neighboring lifecycle and parity regressions**

```bash
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_side_effects.py \
  tldw_Server_API/tests/Jobs/test_jobs_complete_fail_transaction_boundaries.py \
  tldw_Server_API/tests/Jobs/test_jobs_finalize_cancelled_transaction_boundaries.py \
  -q

TLDW_TEST_POSTGRES_REQUIRED=1 RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py \
  -q
```

Expected: all pass with zero PostgreSQL skips.

- [ ] **Step 4: Run static and security gates**

```bash
python -m ruff check \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/operations \
  tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_batch_renew_routing.py \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py \
  tldw_Server_API/tests/Jobs/property/test_operation_contract_properties.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py

python -m compileall -q \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/operations

python -m bandit -r \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/operations/contracts.py \
  tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py \
  tldw_Server_API/app/core/Jobs/operations/postgres/lifecycle.py \
  -f json -o /tmp/bandit_task_12989.json
```

Expected: Ruff and compileall exit zero; Bandit reports no new finding in touched production code.

- [ ] **Step 5: Run diff and boundary checks**

```bash
git diff --check origin/dev...HEAD
rg -n "JobManager|Jobs\.manager|from .*manager|import .*manager" tldw_Server_API/app/core/Jobs/operations
git diff --stat origin/dev...HEAD
git status --short
```

Expected: diff check is clean; the boundary scan finds no `JobManager` dependency; only task-scoped files are changed; worktree is clean after commits.

- [ ] **Step 6: Request independent whole-branch code review**

Use `superpowers:requesting-code-review` against `origin/dev...HEAD`. Review transaction ownership, PostgreSQL RLS cursor use, empty-batch clock behavior, connection/error precedence, SQL parameterization, result invariants, and test durability. Validate each finding with `superpowers:receiving-code-review` before changing code.

- [ ] **Step 7: Record verification and prepare the pull request**

Update `TASK-13010` through Backlog tooling with changed files, exact pass/skip counts, Bandit output location, review findings, commit IDs, and the PR URL. Ask the requester for the mandatory human-authored `Change summary`; do not merge without it.

If task metadata changes after verification, commit it separately:

```bash
git add -A -- \
  "backlog/tasks/task-12989 - Extract-Jobs-batch-lease-renewal-operations-atomically.md" \
  "backlog/tasks/task-13010 - Extract-Jobs-batch-lease-renewal-operations-atomically.md"
git commit -m "chore(jobs): record batch renewal verification"
```
