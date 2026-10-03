# Jobs Completion Row-Identity Atomicity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `JobManager.complete_job` transition only the durable job row established by its locked lookup, with counters and completion outbox state committed atomically.

**Architecture:** Keep `JobManager` as the public facade for this focused defect fix. PostgreSQL locks the visible row with `SELECT ... FOR UPDATE`; SQLite starts `BEGIN IMMEDIATE` before loading it. Both backends capture the raw stored UUID, apply null-safe `id` plus UUID guards to every completion mutation and replay lookup, and return immediately on a locked miss. `WorkerSDK` adds the acquired UUID as an optional caller precondition.

**Tech Stack:** Python 3.11+, FastAPI project runtime, SQLite, PostgreSQL/psycopg 3, pytest, pytest-asyncio, Loguru, Ruff, Black, Bandit, Backlog.md.

---

## Source Of Truth

- Design: `Docs/superpowers/specs/2026-09-07-jobs-completion-row-identity-atomicity-design.md`
- Tracking: `TASK-13215`
- ADR check (2026-10-02): required yes; governing decision:
  `Docs/ADR/058-jobs-completion-row-identity.md`, recording the approved optional
  UUID precondition and locked completion transaction contract.
- Follow-ups excluded from this implementation:
  - `TASK-13216`: migrate other acquired-job callers to `expected_uuid`
  - `TASK-13217`: assess historical completion bookkeeping drift

## Scope Guard

This plan modifies the existing completion facade. It does not extract backend
operations, migrate all direct callers, change schemas, repair historical data,
alter the queued-domain allowlist, or redesign result serialization and SLA
policy lookup. Any defect discovered outside this boundary is recorded in a
separate Backlog task before work continues on it.

Whole-file Black is also outside this defect scope: on the planning baseline,
every existing production/test file listed in Task 6.3 already requires Black
reformatting. Black therefore gates the newly created test module, while Ruff,
syntax checks, `git diff --check`, and changed-hunk review gate all modified
existing files without introducing unrelated formatting churn.

## File Map

- Create: `tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py`
  - Deterministic SQLite and normal PostgreSQL missing-row/replacement races.
  - Expected UUID, legacy UUID, queued-token, and preflight precedence coverage.
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_rls_postgres.py`
  - Forced-RLS visible, hidden, and initial-miss/concurrent-insert completion cases.
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py`
  - Exact counter assertion for simultaneous same-token completion replay.
- Modify: `tldw_Server_API/tests/Jobs/test_worker_sdk.py`
  - Strict proof that ordinary success forwards the acquired UUID.
- Modify: `tldw_Server_API/app/core/Jobs/manager.py:7180`
  - Optional `expected_uuid`, locked authoritative load, direct state branching,
    null-safe identity guards, and base-fact bookkeeping.
- Modify: `tldw_Server_API/app/core/Jobs/worker_sdk.py:1019`
  - Pass `expected_uuid` on ordinary successful completion.
- Modify: `backlog/tasks/task-13215 - Fix-Jobs-completion-missing-row-concurrent-insert-atomicity.md`
  - Record red/green evidence, verification, touched files, and final outcome.
- Modify: this plan
  - Mark stages and task checkboxes as execution proceeds.

## Stage 1: Refresh And Characterize

**Goal:** Start from the live `dev` tip and preserve a passing focused baseline.

**Success Criteria:** The branch is rebased on the exact remote `dev` commit;
existing focused SQLite and required PostgreSQL tests pass before new tests are
added.

**Tests:** Existing completion, lifecycle, transaction-boundary, RLS, and
WorkerSDK suites listed below.

**Status:** Complete

### Task 1: Refresh the branch and record the baseline

**Files:**
- Modify: `backlog/tasks/task-13215 - Fix-Jobs-completion-missing-row-concurrent-insert-atomicity.md`
- Modify: `Docs/superpowers/plans/2026-09-07-jobs-completion-row-identity-atomicity-implementation-plan.md`

- [x] **Step 1: Verify the live remote `dev` tip explicitly**

Run:

```bash
git ls-remote origin refs/heads/dev
git fetch origin +refs/heads/dev:refs/remotes/origin/dev
git rev-parse origin/dev
```

Expected: the `ls-remote` and final `rev-parse` hashes are identical.

- [x] **Step 2: Rebase the isolated branch before implementation**

Run:

```bash
git rebase origin/dev
git merge-base --is-ancestor origin/dev HEAD
git status --short --branch
```

Expected: rebase succeeds, the ancestry check exits zero, and only the plan or
Backlog tracking files intentionally edited during execution are shown.

- [x] **Step 3: Run the focused SQLite baseline**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
RUN_JOBS=1 python -m pytest -q \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py \
  tldw_Server_API/tests/Jobs/test_jobs_complete_fail_transaction_boundaries.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_event_atomicity.py \
  tldw_Server_API/tests/Jobs/test_jobs_finalize_idempotency_sqlite.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk.py \
  -m "not pg_jobs"
```

Expected: exit zero with no failures or errors.

- [x] **Step 4: Run the focused real-PostgreSQL baseline**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
TLDW_TEST_POSTGRES_REQUIRED=1 RUN_JOBS=1 python -m pytest -q \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py \
  tldw_Server_API/tests/Jobs/test_jobs_complete_fail_transaction_boundaries.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_event_atomicity.py \
  tldw_Server_API/tests/Jobs/test_jobs_finalize_idempotency_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_rls_postgres.py \
  -m pg_jobs
```

Expected: exit zero; PostgreSQL tests execute rather than skip. If provisioning
fails, stop and fix the environment rather than weakening the required flag.

- [x] **Step 5: Record exact baseline evidence**

Use the Backlog MCP task edit operation for `TASK-13215` to append the remote
base hash, commands, pass counts, deselections, and warnings. Mark Stage 1
`Complete` and Stage 2 `In Progress` in this plan.

- [x] **Step 6: Commit the refreshed planning checkpoint if tracking changed**

Execution adjustment: consolidate this documentation checkpoint with Task 7.3
after final verification; no implementation checkpoint will contain red tests.

```bash
git add Docs/superpowers/plans/2026-09-07-jobs-completion-row-identity-atomicity-implementation-plan.md \
  "backlog/tasks/task-13215 - Fix-Jobs-completion-missing-row-concurrent-insert-atomicity.md"
git diff --cached --check
git commit -m "docs(jobs): record completion atomicity baseline"
```

Expected: the commit contains only plan/Backlog evidence and passes hooks.

## Stage 2: Prove The Defect And Contract

**Goal:** Add deterministic red tests for both databases and the complete
row-identity contract before production edits.

**Success Criteria:** The missing-row and replacement tests fail for the
validated reason on the unchanged implementation; non-race characterization
tests either pass or identify a separately recorded defect.

**Tests:** New focused module plus forced-RLS additions.

**Status:** Complete

### Task 2: Add deterministic SQLite and PostgreSQL race harnesses

**Files:**
- Create: `tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py`

- [x] **Step 1: Add backend-neutral completion-select detection and hook adapters**

Create the file with these imports and adapters. The hook fires once, after the
database has produced the row but before `complete_job` can classify it:

```python
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager


def _is_completion_select(sql: Any) -> bool:
    normalized = " ".join(str(sql).upper().split())
    return (
        normalized.startswith("SELECT ")
        and "COMPLETION_TOKEN" in normalized
        and "DOMAIN" in normalized
        and "FROM JOBS WHERE ID" in normalized
    )


class _SingleRowCursor:
    def __init__(self, row: Any) -> None:
        self._row = row

    def fetchone(self) -> Any:
        row, self._row = self._row, None
        return row


class _SQLiteCompletionHookConnection:
    def __init__(self, inner: Any, after_fetch: Callable[[], None]) -> None:
        self._inner = inner
        self._after_fetch = after_fetch
        self.fired = False

    def __enter__(self) -> _SQLiteCompletionHookConnection:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: Any, params: Any = ()) -> Any:
        cursor = self._inner.execute(sql, params)
        if not self.fired and _is_completion_select(sql):
            self.fired = True
            row = cursor.fetchone()
            cursor.close()
            self._after_fetch()
            return _SingleRowCursor(row)
        return cursor

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _PostgresCompletionHookCursor:
    def __init__(self, inner: Any, after_fetch: Callable[[Any, bool], None]) -> None:
        self._inner = inner
        self._after_fetch = after_fetch
        self._armed = False
        self.locked = False
        self.fired = False

    def execute(self, sql: Any, params: Any = None) -> Any:
        result = self._inner.execute(sql, params)
        if not self.fired and _is_completion_select(sql):
            normalized = " ".join(str(sql).upper().split())
            self._armed = True
            self.locked = "FOR UPDATE" in normalized
            self.fired = True
        return result

    def fetchone(self) -> Any:
        row = self._inner.fetchone()
        if self._armed:
            self._armed = False
            self._after_fetch(row, self.locked)
        return row

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
```

- [x] **Step 2: Add zero-timeout SQLite helpers and durable snapshots**

```python
def _zero_timeout_sqlite_connection(path: Any) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=0)
    configure_sqlite_connection(conn, busy_timeout_ms=0)
    conn.row_factory = sqlite3.Row
    return conn


def _sqlite_lock_code(exc: sqlite3.OperationalError) -> int | None:
    code = getattr(exc, "sqlite_errorcode", None)
    return None if code is None else int(code) & 0xFF


def _sqlite_counter(jm: JobManager) -> tuple[int, int, int]:
    conn = jm._connect()
    try:
        row = conn.execute(
            "SELECT ready_count, scheduled_count, processing_count "
            "FROM job_counters WHERE domain='chatbooks' "
            "AND queue='default' AND job_type='atomicity'"
        ).fetchone()
        assert row is not None
        return int(row[0]), int(row[1]), int(row[2])
    finally:
        conn.close()


def _sqlite_event_types(jm: JobManager, job_id: int) -> list[str]:
    conn = jm._connect()
    try:
        rows = conn.execute(
            "SELECT event_type FROM job_events WHERE job_id=? ORDER BY id",
            (job_id,),
        ).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        conn.close()
```

- [x] **Step 3: Write the failing SQLite initial-miss race**

Use an empty database so both the absent target and the first concurrent create
have numeric ID 1. The callback captures a zero-timeout busy error instead of
letting it abort completion:

```python
@pytest.mark.unit
@pytest.mark.concurrent
def test_sqlite_locked_miss_cannot_complete_concurrent_insert(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    db_path = tmp_path / "completion-missing-row.db"
    manager = JobManager(db_path)
    inserter = JobManager(db_path)
    monkeypatch.setattr(
        inserter,
        "_connect",
        lambda: _zero_timeout_sqlite_connection(db_path),
    )
    outcome: dict[str, Any] = {}

    def insert_after_missing_fetch() -> None:
        try:
            outcome["job"] = inserter.create_job(
                domain="chatbooks",
                queue="default",
                job_type="atomicity",
                payload={},
                owner_user_id="u1",
            )
        except sqlite3.OperationalError as exc:
            outcome["error"] = exc

    original_connect = manager._connect
    hooked: list[_SQLiteCompletionHookConnection] = []

    def connect_with_hook() -> Any:
        wrapper = _SQLiteCompletionHookConnection(
            original_connect(),
            insert_after_missing_fetch,
        )
        hooked.append(wrapper)
        return wrapper

    monkeypatch.setattr(manager, "_connect", connect_with_hook)

    assert manager.complete_job(1, result={"ok": True}, enforce=False) is False
    assert hooked[0].fired is True
    error = outcome.get("error")
    assert isinstance(error, sqlite3.OperationalError)
    assert _sqlite_lock_code(error) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}

    created = inserter.create_job(
        domain="chatbooks",
        queue="default",
        job_type="atomicity",
        payload={},
        owner_user_id="u1",
    )
    assert int(created["id"]) == 1
    assert manager.get_job(1)["status"] == "queued"
    assert _sqlite_counter(manager) == (1, 0, 0)
    assert _sqlite_event_types(manager, 1) == ["job.created"]
```

- [x] **Step 4: Run the SQLite miss test and preserve red evidence**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
RUN_JOBS=1 python -m pytest -q \
  tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py::test_sqlite_locked_miss_cannot_complete_concurrent_insert
```

Expected before the fix: FAIL because current `complete_job` returns `True`,
the concurrent insert succeeds, and that newly inserted row becomes completed.
Append the exact assertion and durable counter/event snapshot to `TASK-13215`.

- [x] **Step 5: Add the SQLite delete/reinsert identity test**

Create one queued `chatbooks` job. In the completion-select callback, use a raw
zero-timeout connection to delete its event and row and insert a queued row
with the same ID but UUID `replacement-uuid`. Call `complete_job` with
`enforce=False` and the original UUID. Assert after the fix:

```python
assert completion is True
assert isinstance(replacement_attempt.get("error"), sqlite3.OperationalError)
assert _sqlite_lock_code(replacement_attempt["error"]) in {
    sqlite3.SQLITE_BUSY,
    sqlite3.SQLITE_LOCKED,
}
stored = manager.get_job(job_id)
assert stored is not None
assert stored["uuid"] == original["uuid"]
assert stored["status"] == "completed"
assert _sqlite_counter(manager) == (0, 0, 0)
assert _sqlite_event_types(manager, job_id) == ["job.created", "job.completed"]
```

The callback's replacement SQL must bind every value and use the complete
required projection:

```sql
DELETE FROM job_events WHERE job_id = ?;
DELETE FROM jobs WHERE id = ?;
INSERT INTO jobs(
    id, uuid, domain, queue, job_type, owner_user_id,
    payload, status, priority, max_retries, created_at
) VALUES (?, ?, 'chatbooks', 'default', 'atomicity', 'u1',
          '{}', 'queued', 5, 3, DATETIME('now'));
```

Expected before the fix: FAIL because the callback replaces the row and the
ID-only queued update completes `replacement-uuid`.

- [x] **Step 6: Add the normal PostgreSQL missing-row and existing-row lock tests**

Use `jobs_pg_dsn`, two real `JobManager` instances, and a contextmanager around
the target manager's original `_pg_cursor`. For the missing-row callback,
create the first `chatbooks` job from the second manager and assert the hook
reported `row is None`. For the existing-row callback, use a separate psycopg
connection and run:

```sql
SELECT id FROM jobs WHERE id = %s FOR UPDATE NOWAIT
```

Capture `psycopg.errors.LockNotAvailable` and roll back that competing
transaction. Final assertions:

```python
assert manager.complete_job(1, result={"ok": True}, enforce=False) is False
inserted = inserter.get_job(1)
assert inserted is not None
assert inserted["status"] == "queued"
assert postgres_counter(inserter, "chatbooks", "default", "atomicity") == (1, 0, 0)
assert postgres_event_types(inserter, 1) == ["job.created"]
assert completion_hook.locked is True
assert isinstance(lock_attempt["error"], psycopg.errors.LockNotAvailable)
```

Implement `postgres_counter` and `postgres_event_types` with the manager's
normal `_pg_cursor` and these bound reads:

```sql
SELECT ready_count, scheduled_count, processing_count
FROM job_counters
WHERE domain=%s AND queue=%s AND job_type=%s;

SELECT event_type
FROM job_events
WHERE job_id=%s
ORDER BY id;
```

Use `try/finally` to close every competing connection. Mark both tests
`pytest.mark.pg_jobs` and `pytest.mark.concurrent`.
Import psycopg inside each PostgreSQL test with
`psycopg = pytest.importorskip("psycopg")` so a minimal SQLite environment does
not skip the complete module.

- [x] **Step 7: Run the PostgreSQL race tests and preserve red evidence**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
TLDW_TEST_POSTGRES_REQUIRED=1 RUN_JOBS=1 python -m pytest -q \
  tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py \
  -m pg_jobs
```

Expected before the fix: the missing-row test fails because the concurrent row
is completed; the existing-row test fails because the initial SELECT lacks
`FOR UPDATE` and the competing NOWAIT lock succeeds.

Do not commit while these tests are red.

### Task 3: Add RLS, expected-UUID, state, and precedence tests

**Files:**
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_rls_postgres.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py`

- [x] **Step 1: Add expected-UUID and legacy-value tests for both backends**

Use the established backend parametrization pattern:

```python
_BACKENDS = (
    "sqlite",
    pytest.param("postgres", marks=pytest.mark.pg_jobs),
)


def _manager_for_backend(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    suffix: str,
) -> JobManager:
    if backend == "postgres":
        return JobManager(
            None,
            backend="postgres",
            db_url=request.getfixturevalue("jobs_pg_dsn"),
        )
    return JobManager(tmp_path / f"completion-{suffix}.db")
```

Add tests that create and acquire a job, then assert:

```python
assert manager.complete_job(
    int(acquired["id"]),
    worker_id=str(acquired["worker_id"]),
    lease_id=str(acquired["lease_id"]),
    completion_token=str(acquired["lease_id"]),
    expected_uuid="different-incarnation",
    enforce=True,
) is False
assert manager.get_job(int(acquired["id"]))["status"] == "processing"
```

For legacy coverage, update the acquired row UUID to each value below through
a backend-specific bound statement, then complete with the corresponding
external representation:

```python
pytest.param(None, "", id="null-uuid")
pytest.param("legacy uuid value", "legacy uuid value", id="noncanonical-uuid")
```

Assert completion succeeds and the stored UUID is not rewritten.
Repeat completion with the same token and expected UUID, assert it returns
`True`, and verify the lifecycle counter remains terminal and exactly one
`job.completed` event exists.

- [x] **Step 2: Add queued completion-token guard coverage**

Create an allowlisted queued job, set `completion_token='token-a'`, and call
with `completion_token='token-b'` and `enforce=False`. This characterization
intentionally omits `expected_uuid` so it isolates the existing queued-token
contract before the new keyword exists:

```python
assert completed is False
stored = manager.get_job(job_id)
assert stored is not None
assert stored["status"] == "queued"
assert stored["completion_token"] == "token-a"
```

Then call with `token-a` and assert one successful transition, one counter
delta, and one `job.completed` event.

- [x] **Step 3: Add preflight precedence tests without freezing serialization defects**

On SQLite, call a missing ID and an existing stale-UUID row while varying only
the documented preflight error:

```python
monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "true")
with pytest.raises(ValueError, match="completion_token required"):
    manager.complete_job(missing_id, expected_uuid="stale")

monkeypatch.delenv("JOBS_REQUIRE_COMPLETION_TOKEN", raising=False)
monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "4")
with pytest.raises(ValueError, match="Result too large"):
    manager.complete_job(missing_id, result={"value": "too large"})

monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "invalid")
with pytest.raises(ValueError):
    manager.complete_job(missing_id, expected_uuid="stale")
```

Do not add a nonserializable-result test.

- [x] **Step 4: Add forced-RLS completion tests**

In `test_jobs_rls_postgres.py`, reuse `_dsn_or_skip`,
`ensure_jobs_rls_policies_pg`, `_seed_processing_job`, and
`_read_lifecycle_facts`. Add `from contextlib import contextmanager` and use it
for the narrowly scoped `_pg_cursor` hook; do not replace or bypass the RLS
setup performed by the manager's original cursor context.

Visible legacy-null row:

```python
JobManager.set_rls_context(
    is_admin=False,
    domain_allowlist="chatbooks",
    owner_user_id="u1",
)
try:
    assert manager.complete_job(
        job_id,
        worker_id="worker-1",
        lease_id="lease-1",
        completion_token="lease-1",
        expected_uuid="",
        enforce=True,
    ) is True
finally:
    JobManager.clear_rls_context()
```

Hidden `u2` row under the `u1` context must return `False` and preserve the
complete `_read_lifecycle_facts` tuple.

For the RLS initial-miss race, wrap only the completion cursor's first
completion SELECT. After it returns no row, insert an explicit visible
`chatbooks`/`u1` queued row with the target ID through the admin DSN and commit.
Assert completion returns `False` and the admin read still sees `queued`.
Always clear RLS context and close the competing connection in `finally`.

- [x] **Step 5: Strengthen concurrent same-token side-effect coverage**

In
`test_simultaneous_same_operation_finalizers_emit_one_durable_event`, retain
the existing event assertion and add:

```python
if operation == "complete":
    assert _counter_snapshot(manager, acquired) == (0, 0, 0, 0)
```

This proves the `True, True` applied-plus-replay result creates one durable
event and one processing-counter decrement.

- [x] **Step 6: Prove SLA savepoint-control failures remain transaction-fatal**

In `test_jobs_lifecycle_hardening_regressions.py`, add a connection/cursor
adapter that replaces only `SAVEPOINT job_completion_sla` with a statement
against `missing_jobs_sla_control`. Configure an enabled duration SLA, backdate
the acquired job, and patch the target manager's `_connect` to return the
adapter:

```python
if backend == "postgres":
    import psycopg

    expected_error = psycopg.Error
else:
    expected_error = sqlite3.Error

with pytest.raises(expected_error):
    manager.complete_job(
        int(acquired["id"]),
        worker_id=str(acquired["worker_id"]),
        lease_id=str(acquired["lease_id"]),
        completion_token="token-a",
        enforce=True,
    )

reader = _clone_manager(manager)
stored = reader.get_job(int(acquired["id"]))
assert stored is not None
assert stored["status"] == "processing"
assert _counter_snapshot(reader, acquired) == (0, 0, 1, 0)
assert _event_count(reader, int(acquired["id"]), "job.completed") == 0
assert _attachment_count(reader, int(acquired["id"])) == 0
```

For PostgreSQL, the cursor adapter calls
`inner.execute("SELECT * FROM missing_jobs_sla_control")`; for SQLite, the
connection adapter does the same through `inner.execute`. Delegate transaction
entry/exit, commit, rollback, close, and all non-target statements unchanged.

- [x] **Step 7: Run the new contract tests before implementation**

Run the SQLite subset:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
RUN_JOBS=1 python -m pytest -q \
  tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py \
  -m "not pg_jobs"
```

Run the required PostgreSQL subset:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
TLDW_TEST_POSTGRES_REQUIRED=1 RUN_JOBS=1 python -m pytest -q \
  tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py \
  tldw_Server_API/tests/Jobs/test_jobs_rls_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py \
  -m pg_jobs
```

Expected: expected-UUID tests raise `TypeError` because the keyword does not yet
exist; both missing-row races and lock assertions fail for the validated
reason. Existing state/preflight characterization remains green.

Do not commit while the required regression tests are red.

## Stage 3: Implement The Atomic Boundary

**Goal:** Make both backend paths satisfy the same locked-row contract while
preserving existing facade behavior and bookkeeping.

**Success Criteria:** Every Stage 2 test is green; existing completion,
transaction, event, counter, SLA, and idempotency tests remain green.

**Tests:** Stage 2 suites plus existing focused regressions.

**Status:** Complete

### Task 4: Implement locked identity in `JobManager.complete_job`

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/manager.py:7180-7785`
- Test: `tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py`
- Test: `tldw_Server_API/tests/Jobs/test_jobs_rls_postgres.py`
- Test: `tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py`

- [x] **Step 1: Add the optional public precondition without changing earlier validation order**

Add the keyword after `completion_token` and before `enforce`:

```python
def complete_job(
    self,
    job_id: int,
    *,
    result: dict[str, Any] | None = None,
    worker_id: str | None = None,
    lease_id: str | None = None,
    completion_token: str | None = None,
    expected_uuid: str | None = None,
    enforce: bool | None = None,
) -> bool:
```

Do not normalize or validate `expected_uuid` before required-token,
enforcement, maximum-size parsing, and result-size checks.

- [x] **Step 2: Replace PostgreSQL's initial read with the authoritative lock**

Use one projection containing all mutation and side-effect facts:

```python
cur.execute(
    "SELECT id, uuid, status, completion_token, worker_id, lease_id, "
    "domain, queue, job_type, available_at, started_at, acquired_at, "
    "trace_id, request_id, owner_user_id "
    "FROM jobs WHERE id = %s FOR UPDATE",
    (int(job_id),),
)
base = cur.fetchone()
if base is None:
    return False
stored_uuid = base.get("uuid")
if (
    expected_uuid is not None
    and str(stored_uuid or "") != str(expected_uuid)
):
    return False
```

Classify terminal state immediately from this row. Preserve exact completed
token replay and return `False` for failed, cancelled, quarantined, and
mismatched completed states.

- [x] **Step 3: Branch directly from the locked PostgreSQL state**

After domain-scoped encryption, serialize the final result once and initialize
the transition flags:

```python
result_json = json.dumps(res_obj) if res_obj is not None else None
completed_from_processing = False
completed_from_queued = False
ok = False
```

For enforced processing, add the captured UUID to the current ownership and
token guards:

```sql
UPDATE jobs
SET status = 'completed', result = %s::jsonb, completed_at = NOW(),
    completion_token = %s, leased_until = NULL,
    worker_id = NULL, lease_id = NULL
WHERE id = %s
  AND uuid IS NOT DISTINCT FROM %s
  AND status = 'processing'
  AND worker_id = %s
  AND lease_id = %s
  AND (completion_token IS NULL OR completion_token = %s)
```

For unenforced processing, retain the token guard and remove only worker/lease
guards. For queued state, compute the existing allowlist once from the locked
`base["domain"]`; only issue the queued update when enforcement is disabled,
the domain is allowlisted, and the same token guard is present. Both statements
must include:

```sql
id = %s AND uuid IS NOT DISTINCT FROM %s
```

Do not issue the current secondary `SELECT domain FROM jobs`.

- [x] **Step 4: Scope PostgreSQL's defensive replay to the captured identity**

If the selected branch's update affects zero rows and a token was supplied,
use:

```python
cur.execute(
    "SELECT completion_token, status FROM jobs "
    "WHERE id = %s AND uuid IS NOT DISTINCT FROM %s",
    (int(job_id), stored_uuid),
)
replay = cur.fetchone()
if (
    replay
    and str(replay.get("status") or "") == "completed"
    and replay.get("completion_token")
    and str(replay["completion_token"]) == str(completion_token)
):
    return True
```

Otherwise leave `ok=False`. Keep counter reconciliation, SLA savepoint work,
completion outbox insertion, explicit commit, and post-commit callbacks under
`if ok`, using the authoritative `base` facts.

- [x] **Step 5: Add SQLite's writer boundary and named authoritative row**

At the start of the SQLite transaction, before the first job query:

```python
with conn:
    conn.execute("BEGIN IMMEDIATE")
    row = conn.execute(
        "SELECT id, uuid, status, completion_token, worker_id, lease_id, "
        "domain, queue, job_type, available_at, started_at, acquired_at, "
        "trace_id, request_id, owner_user_id "
        "FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    if row is None:
        return False
    base = dict(row)
    stored_uuid = base.get("uuid")
    if (
        expected_uuid is not None
        and str(stored_uuid or "") != str(expected_uuid)
    ):
        return False
```

Use `base` by field name throughout SQLite metrics, counters, SLA, and event
construction. Do not retain positional `rowm[...]` accesses after adding UUID
to the projection.

- [x] **Step 6: Apply SQLite's direct state branches and null-safe guards**

Use the same state decisions as PostgreSQL. Every SQLite update and defensive
replay uses:

```sql
id = ? AND uuid IS ?
```

The enforced processing parameter order is:

```python
(
    result_json,
    completion_token,
    job_id,
    stored_uuid,
    worker_id,
    lease_id,
    completion_token,
)
```

The queued branch reads only `base["domain"]` for the existing allowlist and
must not query the jobs table again. Preserve `SELECT changes()` row-count
handling unless a characterization test proves a backend issue.

- [x] **Step 7: Keep mandatory and optional transaction failures distinct**

Do not add exception suppression around counter updates, reconciliation,
`job.completed` insertion, savepoint creation/release, or commit. Retain the
existing suppression inside `_stage_completion_sla_breach` only for statement
failures after its savepoint exists. Queue and run observers only after a
successful commit.

- [x] **Step 8: Run the complete red-to-green matrix**

SQLite:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
RUN_JOBS=1 python -m pytest -q \
  tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py \
  tldw_Server_API/tests/Jobs/test_jobs_complete_fail_transaction_boundaries.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_event_atomicity.py \
  tldw_Server_API/tests/Jobs/test_jobs_finalize_idempotency_sqlite.py \
  -m "not pg_jobs"
```

PostgreSQL:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
TLDW_TEST_POSTGRES_REQUIRED=1 RUN_JOBS=1 python -m pytest -q \
  tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py \
  tldw_Server_API/tests/Jobs/test_jobs_rls_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py \
  tldw_Server_API/tests/Jobs/test_jobs_complete_fail_transaction_boundaries.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_event_atomicity.py \
  tldw_Server_API/tests/Jobs/test_jobs_finalize_idempotency_postgres.py \
  -m pg_jobs
```

Expected: all selected tests pass; required PostgreSQL tests do not skip.

- [x] **Step 9: Review the diff for forbidden scope expansion**

Run:

```bash
git diff -- tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py \
  tldw_Server_API/tests/Jobs/test_jobs_rls_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py
```

Expected: no schema, operation-module, unrelated lifecycle, serialization, or
historical-repair changes.

- [x] **Step 10: Commit the green manager boundary**

```bash
git add tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py \
  tldw_Server_API/tests/Jobs/test_jobs_rls_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py \
  Docs/superpowers/plans/2026-09-07-jobs-completion-row-identity-atomicity-implementation-plan.md \
  "backlog/tasks/task-13215 - Fix-Jobs-completion-missing-row-concurrent-insert-atomicity.md"
git diff --cached --check
git commit -m "fix(jobs): bind completion to locked row identity"
```

Expected: hooks pass and the commit contains the complete cross-backend
contract, its tests, and current execution evidence.

## Stage 4: Bind Ordinary Workers To Acquired Identity

**Goal:** Give `WorkerSDK` pre-call stale-row rejection without expanding the
change to other direct callers.

**Success Criteria:** Ordinary successful completion passes the acquired UUID
through the new keyword and does not retry without it.

**Tests:** Focused async WorkerSDK success and callback tests.

**Status:** Complete

### Task 5: Add and implement strict WorkerSDK UUID forwarding

**Files:**
- Modify: `tldw_Server_API/tests/Jobs/test_worker_sdk.py`
- Modify: `tldw_Server_API/app/core/Jobs/worker_sdk.py:1019-1031`

- [x] **Step 1: Make the existing success spy enforce the exact keyword contract**

In `test_run_success_completes_job`, replace the permissive `**kwargs` spy with
an explicit keyword-only signature:

Add `from typing import Any` to the test module imports, then use:

```python
def spy_complete(
    job_id: int,
    *,
    result: dict[str, Any] | None,
    worker_id: str | None,
    lease_id: str | None,
    completion_token: str | None,
    expected_uuid: str | None,
    enforce: bool | None,
) -> bool:
    calls.append(
        {
            "job_id": job_id,
            "result": result,
            "worker_id": worker_id,
            "lease_id": lease_id,
            "completion_token": completion_token,
            "expected_uuid": expected_uuid,
            "enforce": enforce,
        }
    )
    return orig_complete(
        job_id,
        result=result,
        worker_id=worker_id,
        lease_id=lease_id,
        completion_token=completion_token,
        expected_uuid=expected_uuid,
        enforce=enforce,
    )
```

Add:

```python
assert calls[0]["expected_uuid"] == job["uuid"]
```

- [x] **Step 2: Run the strict WorkerSDK test red**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
RUN_JOBS=1 python -m pytest -q \
  tldw_Server_API/tests/Jobs/test_worker_sdk.py::test_run_success_completes_job
```

Expected before the WorkerSDK edit: FAIL because the strict spy requires
`expected_uuid`; the worker must not silently recover by dropping it.

- [x] **Step 3: Pass the acquired UUID from the ordinary success path**

Add exactly one argument to the existing `self.jm.complete_job` call:

```python
expected_uuid=str(job.get("uuid") or ""),
```

Do not change `WorkerTerminalOutcome`, cancellation, failure, callback, or
direct service-worker paths in this task.

- [x] **Step 4: Run WorkerSDK tests green**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
RUN_JOBS=1 python -m pytest -q \
  tldw_Server_API/tests/Jobs/test_worker_sdk.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py
```

Expected: all selected tests pass.

- [x] **Step 5: Commit WorkerSDK forwarding**

```bash
git add tldw_Server_API/app/core/Jobs/worker_sdk.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk.py \
  Docs/superpowers/plans/2026-09-07-jobs-completion-row-identity-atomicity-implementation-plan.md \
  "backlog/tasks/task-13215 - Fix-Jobs-completion-missing-row-concurrent-insert-atomicity.md"
git diff --cached --check
git commit -m "fix(jobs): bind worker completion to acquired UUID"
```

Expected: hooks pass; no direct caller outside `WorkerSDK` changes.

## Stage 5: Verify, Review, And Prepare Integration

**Goal:** Prove the focused fix, check security and style, and leave a clean
reviewable branch with complete task evidence.

**Success Criteria:** Focused and broader Jobs tests pass on SQLite and required
PostgreSQL; applicable format, lint, compile, and Bandit checks pass; review
finds no unresolved in-scope issue; task evidence is current.

**Tests:** Full commands below plus mandatory Jobs CI after PR creation.

**Status:** Complete

Local implementation, review, verification, final rebase, and draft PR
preparation are complete. Remote CI and the human-authored `Change summary`
remain merge gates; this stage does not claim that CI passed or authorize merge.

### Task 6: Run full local verification and security checks

**Files:**
- Verify: `tldw_Server_API/app/core/Jobs/manager.py`
- Verify: `tldw_Server_API/app/core/Jobs/worker_sdk.py`
- Verify: `tldw_Server_API/tests/Jobs`

- [x] **Step 1: Run the full focused SQLite matrix fresh**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
RUN_JOBS=1 python -m pytest -q \
  tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py \
  tldw_Server_API/tests/Jobs/test_jobs_complete_fail_transaction_boundaries.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_event_atomicity.py \
  tldw_Server_API/tests/Jobs/test_jobs_finalize_idempotency_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_admin_counters_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_events_outbox_sqlite.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py \
  -m "not pg_jobs"
```

Expected: exit zero with no failures or errors.

- [x] **Step 2: Run the full focused required-PostgreSQL matrix fresh**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
TLDW_TEST_POSTGRES_REQUIRED=1 RUN_JOBS=1 python -m pytest -q \
  tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py \
  tldw_Server_API/tests/Jobs/test_jobs_rls_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py \
  tldw_Server_API/tests/Jobs/test_jobs_complete_fail_transaction_boundaries.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_event_atomicity.py \
  tldw_Server_API/tests/Jobs/test_jobs_finalize_idempotency_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_admin_counters_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_events_outbox_postgres.py \
  -m pg_jobs
```

Set `RUN_PG_JOBS_TESTS=1` and `JOBS_SSE_TEST_MAX_SECONDS=0.5` for the existing
opt-in outbox/SSE tests. Once the shared fixture's container is running,
`TLDW_TEST_NO_DOCKER=1` avoids redundant provisioning.

Expected: exit zero and no PostgreSQL test skips.

- [x] **Step 3: Run formatting, lint, and syntax checks**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m black --check \
  tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py
python -m ruff check \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/worker_sdk.py \
  tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py \
  tldw_Server_API/tests/Jobs/test_jobs_rls_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk.py
python -m py_compile \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/worker_sdk.py
```

Expected: every command exits zero. Black checks the newly created module;
Ruff and syntax checks cover all touched runtime files, and Task 6.5 checks
whitespace across the complete branch diff. Review changed hunks in the
pre-existing files for local formatting consistency without reformatting
untouched code.

- [x] **Step 4: Run scoped Bandit**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/worker_sdk.py \
  -f json -o /tmp/bandit_task_13215.json
```

Expected: the JSON report contains no new findings in changed code. Compare
any finding against the identical scope archived from the base revision and
run Bandit's `--baseline` check; require exit zero and no new results/errors.
Record raw and delta report paths and summary in `TASK-13215`. Do not suppress
an existing finding to claim that the raw scan is clean.

- [x] **Step 5: Inspect final scope and repository state**

```bash
git diff origin/dev...HEAD --check
git diff --stat origin/dev...HEAD
git status --short --branch
```

Expected: only the design/plan/tasks, the two production files, and the focused
Jobs tests are present; the worktree is clean after final tracking updates.

### Task 7: Request review and prepare the branch for a PR

**Files:**
- Modify: `backlog/tasks/task-13215 - Fix-Jobs-completion-missing-row-concurrent-insert-atomicity.md`
- Modify: `Docs/superpowers/plans/2026-09-07-jobs-completion-row-identity-atomicity-implementation-plan.md`

- [x] **Step 1: Perform the required code review**

Invoke `superpowers:requesting-code-review`. Validate every finding against the
current code and tests before editing. For any valid in-scope issue, use
`superpowers:receiving-code-review` and `superpowers:test-driven-development` to
add a failing test, apply the smallest correction, and rerun the affected
matrix. Record unrelated defects in separate Backlog tasks.

- [x] **Step 2: Update execution records**

Mark all completed stages and checkboxes in this plan. Through Backlog MCP,
update `TASK-13215` with:

```text
- exact remote dev base
- red failure evidence and green pass evidence
- SQLite and PostgreSQL/RLS test counts
- formatting, Ruff, syntax, and Bandit results
- review findings and dispositions
- final touched files
- TASK-13216 and TASK-13217 as excluded follow-ups
```

- [x] **Step 3: Commit final review evidence**

```bash
git add Docs/superpowers/plans/2026-09-07-jobs-completion-row-identity-atomicity-implementation-plan.md \
  "backlog/tasks/task-13215 - Fix-Jobs-completion-missing-row-concurrent-insert-atomicity.md"
git diff --cached --check
git commit -m "docs(jobs): record completion atomicity verification"
```

Expected: commit hooks pass and no production or test file is unexpectedly
staged.

- [x] **Step 4: Rebase and rerun affected verification before PR creation**

Refresh the explicit remote `dev` ref and rebase. If `manager.py`,
`worker_sdk.py`, another Jobs runtime dependency, or any touched test changed
upstream, rerun Tasks 6.1 through 6.4 in full. Otherwise, still rerun the new
atomicity module, forced-RLS completion cases, and strict WorkerSDK success test
on both applicable backends. Never resolve unrelated history by replaying old
`dev` commits; verify the remote hash before rebasing.

- [x] **Step 5: Prepare the PR without merging**

Push the branch and create a PR against `dev`. Do not author the required human
`Change summary` on the requester's behalf. The PR remains draft or explicitly
merge-blocked until the requester adds a summary explaining what changed and
why the lock/identity approach was selected. Request the mandatory Jobs CI
checks and leave merge as a separate explicit user action.

Expected PR scope:

```text
Prevent complete_job from transitioning a row inserted or replaced after its
authoritative lookup. Lock and guard the exact stored identity on SQLite and
PostgreSQL, keep enabled counters/outbox atomic, and pass acquired UUIDs from
WorkerSDK. Preserve the public boolean facade and defer broad caller migration
and historical repair to TASK-13216 and TASK-13217.
```

## Execution Record: 2026-10-02

- Rebased the three planning commits onto live dev
  `9958110df2a9011e19f48b0eae821353e19d4af8` before implementation.
- Baselines: SQLite 127 passed / 55 deselected; real PostgreSQL 67 passed /
  56 deselected, including the newly added SLA control characterization.
  Docker Desktop had to be started after the initial fixture setup failures.
- Red evidence on unchanged code: SQLite miss and replacement tests both failed;
  PostgreSQL miss and initial row-lock tests both failed; UUID contract tests
  raised the expected unsupported-keyword errors. Forced RLS reproduced the
  visible-row insertion race (`True` with a newly completed row). Strict worker
  completion failed because the required UUID keyword was absent.
- Implemented both database paths in the existing facade and forwarded the
  acquired UUID from ordinary WorkerSDK success. No operation extraction or
  broader caller migration was performed.
- Expanded green matrices: SQLite 242 passed / 65 deselected; PostgreSQL 83
  passed / 71 deselected, with two pre-existing opt-in SSE/outbox skips.
  Reran those two with `RUN_PG_JOBS_TESTS=1` and the supported bounded-stream
  setting `JOBS_SSE_TEST_MAX_SECONDS=0.5`: 2 passed, no skips.
- WorkerSDK and prepared-worker suites: 132 passed. New atomicity module:
  24 passed across SQLite and required PostgreSQL. Test output contains
  existing project warnings; no selected test failure remains.
- Black (new module), Ruff (all six touched Python files), runtime syntax
  compilation, and whitespace checks passed. Scoped Bandit reported one
  unchanged B608 warning in canonical webhook pruning outside this patch;
  comparison against an archive of dev produced no new findings and no errors
  (`/tmp/bandit_task_13215_delta.json`, exit zero).
- Initial spec review found two test weaknesses, both addressed: the hidden-row
  snapshot now includes result/token/completed-time/UUID, and concurrent replay
  retains a second processing job to prove the exact counter change `2 -> 1`.
- The final upstream update to `86e287fee7bfa1a1588639232e35db3666851ded`
  changes only unrelated MCP gateway tests and their task record. Final rebase
  and focused rerun completed before PR creation.
- ADR-058 records the approved durable completion contract under the current
  repository ADR workflow. TASK-13216 and TASK-13217 remain separate follow-ups.
- Final review added forced-RLS competing `NOWAIT` lock proof, missing-counter
  reconciliation SQL-error rollback, both optional SLA write errors, and four
  SLA control failure paths. The spec re-review confirmed every identified
  coverage gap is closed. Runtime/worker quality review found no actionable
  finding; the final lifecycle-test quality review also found no actionable issue.
- Fresh integrated SQLite matrix: 247 passed / 70 deselected / 1212 warnings.
  Ruff, new-module Black, runtime compilation, and the Bandit delta scan passed
  again; the delta report has no results or errors.
- Fresh integrated required PostgreSQL matrix: 90 passed / 76 deselected /
  358 warnings, no skips. The two opt-in SSE tests ran in this matrix.
- Test-scope Bandit (excluding pytest assertion warning B101) uses normalized
  archived filenames and reports no new findings/errors in
  `/tmp/bandit_task_13215_tests_delta.json`. Two new fixture-token warnings were
  removed by reusing the acquired UUID and seeded lease value. Focused reruns
  after that cleanup passed: SQLite 5 and PostgreSQL/RLS 8.
- The green manager/test boundary and WorkerSDK forwarding were committed
  separately, with references to TASK-13215 and this plan. Documentation and
  baseline checkpoints are consolidated into the final verification record.
- Final rebase onto explicitly verified live dev
  `86e287fee7bfa1a1588639232e35db3666851ded` succeeded without conflicts.
  Post-rebase verification: SQLite atomicity/strict-worker 16 passed /
  9 deselected; required PostgreSQL atomicity/forced-RLS 12 passed /
  15 deselected. A mistyped PostgreSQL test node initially failed collection;
  the corrected actual node ran successfully. Ruff and complete branch
  whitespace checks passed again.
- Draft PR: https://github.com/rmusser01/tldw_server/pull/3092, base `dev`.
  Initial `gh pr create` resolved the unrelated upstream repository and failed;
  explicit `--repo rmusser01/tldw_server` created the verified draft without
  altering upstream. Dedicated Jobs CI was queued automatically for the PR.
  Repository-required checks and Jobs CI are pending, not waived.
- The branch/worktree is retained for PR review. TASK-13215 remains in progress
  pending integration. The human requester must write the required
  `Change summary`; no merge was performed. Strict extraction remains a
  separate next work item after this defect fix merges.
