# Jobs Admission Counter Baseline Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconcile the stale cross-backend admission counter regression test with the merged SQLite and PostgreSQL durability contracts, then close the completed renewal/release workstream tracking.

**Architecture:** Keep all production Jobs code unchanged. Replace the shared rollback expectation with explicit backend tests: SQLite counter updates remain transaction-critical, while PostgreSQL counter updates remain best effort inside the existing savepoint and therefore cannot abort durable job/event admission. Record required real-PostgreSQL verification and PR #2763 merge evidence before closing the prior child task and plan stage.

**Tech Stack:** Python 3.14, pytest, sqlite3, psycopg 3, existing Jobs PostgreSQL fixtures, Backlog.md MCP, Ruff, Bandit.

---

## File Structure

- Modify: `tldw_Server_API/tests/Jobs/test_jobs_dependency_acquire_counter_regressions.py`
  - Split one stale cross-backend assertion into explicit SQLite rollback and PostgreSQL best-effort durability contracts.
  - Add a narrow counter-row count helper so PostgreSQL proves the failed optional update was not persisted.
- Modify: `Docs/superpowers/plans/2026-07-14-jobs-admission-hardening-and-lease-lifecycle.md`
  - Mark Stage 5 and Task 13 Step 4 complete with PR #2763 merge evidence and the parent-ID collision note.
- Modify through Backlog MCP: `TASK-12969.3`
  - Add the merge evidence and final summary, check the remaining Definition of Done item, and mark the child task Done.
- Modify through Backlog MCP: `TASK-12988`
  - Track this reconciliation slice, verification evidence, touched files, and the unresolved duplicate parent-ID blocker.

## Stage 1: Reproduce and Lock the Contract
**Goal:** Verify the stale expectation on clean merged `origin/dev` before editing tests.
**Success Criteria:** Both SQLite parameterizations pass and both required PostgreSQL parameterizations fail with `Failed: DID NOT RAISE RuntimeError`.
**Tests:** The existing four-case admission counter failure test with `RUN_JOBS=1` and the real PostgreSQL fixture.
**Status:** Complete

### Task 1: Capture the clean baseline

**Files:**
- Read: `tldw_Server_API/tests/Jobs/test_jobs_dependency_acquire_counter_regressions.py`
- Read: `tldw_Server_API/app/core/Jobs/operations/postgres/admission.py`

- [x] **Step 1: Run the existing four-case regression test**

Run:

```bash
source ../../.venv/bin/activate
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_dependency_acquire_counter_regressions.py::test_admission_counter_failure_rolls_back_job_and_created_event \
  -q -rs
```

Expected: two SQLite passes and two PostgreSQL failures because PostgreSQL commits instead of raising.

- [x] **Step 2: Confirm the production transaction boundary**

Inspect `_bump_counters_best_effort` and verify it rolls counter failures back to `jobs_admission_counter_update`, releases the savepoint, and allows the durable job/event transaction to commit.

## Stage 2: Reconcile Backend-Specific Expectations
**Goal:** Make the regression suite state each backend's merged durability contract directly.
**Success Criteria:** SQLite proves full rollback; PostgreSQL proves durable job/event admission and no counter row after the injected counter failure.
**Tests:** Four focused parameterizations covering plain and idempotent admission for both backends.
**Status:** Complete

### Task 2: Split the stale regression test

**Files:**
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_dependency_acquire_counter_regressions.py`

- [x] **Step 1: Add a backend-neutral counter-row count helper**

Add this parameterized query helper beside `_counter` so a failed optional counter write can be asserted without requiring a row:

```python
def _counter_row_count(manager: JobManager, *, domain: str, job_type: str) -> int:
    conn = manager._connect()
    try:
        if manager.backend == "postgres":
            with manager._pg_cursor(conn) as cur:
                cur.execute(
                    "SELECT COUNT(*) AS count FROM job_counters "
                    "WHERE domain=%s AND queue='default' AND job_type=%s",
                    (domain, job_type),
                )
                return int(cur.fetchone()["count"])
        row = conn.execute(
            "SELECT COUNT(*) FROM job_counters "
            "WHERE domain=? AND queue='default' AND job_type=?",
            (domain, job_type),
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()
```

- [x] **Step 2: Preserve the SQLite rollback contract**

Replace the shared test's SQLite half with:

```python
@pytest.mark.parametrize("idempotency_key", [None, "same"], ids=["plain", "idempotent"])
def test_sqlite_admission_counter_failure_rolls_back_job_and_created_event(
    idempotency_key: str | None,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    manager = _manager("sqlite", request=request, tmp_path=tmp_path, name="admission-counter")
    domain = "admission-counter-failure-sqlite"
    import tldw_Server_API.app.core.Jobs.operations.sqlite.admission as adapter

    def fail_counter(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("forced admission counter failure")

    monkeypatch.setattr(adapter, "_bump_counters", fail_counter)
    with pytest.raises(RuntimeError, match="forced admission counter failure"):
        manager.create_job(
            domain=domain,
            queue="default",
            job_type="work",
            payload={},
            owner_user_id="owner",
            idempotency_key=idempotency_key,
        )
    assert _job_and_event_counts(manager, domain=domain) == (0, 0)
```

- [x] **Step 3: State the PostgreSQL best-effort contract**

Add this `pytest.mark.pg_jobs` test for the same plain/idempotent cases:

```python
@pytest.mark.pg_jobs
@pytest.mark.parametrize("idempotency_key", [None, "same"], ids=["plain", "idempotent"])
def test_postgres_admission_counter_failure_keeps_job_and_created_event(
    idempotency_key: str | None,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    manager = _manager("postgres", request=request, tmp_path=tmp_path, name="admission-counter")
    domain = "admission-counter-failure-postgres"
    import tldw_Server_API.app.core.Jobs.operations.postgres.admission as adapter

    def fail_counter(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("forced admission counter failure")

    monkeypatch.setattr(adapter, "_bump_counters", fail_counter)
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
        idempotency_key=idempotency_key,
    )

    assert job["domain"] == domain
    assert _job_and_event_counts(manager, domain=domain) == (1, 1)
    assert _counter_row_count(manager, domain=domain, job_type="work") == 0
```

- [x] **Step 4: Run the focused test contract**

Run:

```bash
source ../../.venv/bin/activate
RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_dependency_acquire_counter_regressions.py \
  -k admission_counter_failure \
  -q -rs
```

Expected: four passes, zero skips.

Execution evidence: all four cases passed against SQLite and required real PostgreSQL with zero skips. The PostgreSQL cases persisted one job and one durable created event while the injected optional counter write left no `job_counters` row.

## Stage 3: Regression Gates and Tracking Closure
**Goal:** Prove the corrected baseline across neighboring Jobs behavior and close the merged lifecycle stream accurately.
**Success Criteria:** Required PostgreSQL matrices have zero skips; lint/security/diff checks pass; child task and plan carry merge evidence; ambiguous parent records are untouched.
**Tests:** Focused admission suites, the unchanged 104-case neighboring matrix, Ruff, Bandit, and `git diff --check`.
**Status:** Complete

### Task 3: Verify and finalize

**Files:**
- Modify: `Docs/superpowers/plans/2026-07-14-jobs-admission-hardening-and-lease-lifecycle.md`
- Update through Backlog MCP: `TASK-12969.3`
- Update through Backlog MCP: `TASK-12988`

- [x] **Step 1: Run focused admission verification**

Run the SQLite/PostgreSQL admission operation, quota, secret-hygiene, and dependency/counter regression suites with `RUN_JOBS=1`. Expected: all selected tests pass and required PostgreSQL has zero skips.

Execution evidence: 110 admission, quota, secret-hygiene, and dependency/counter tests passed against SQLite and required real PostgreSQL with zero skips.

- [x] **Step 2: Run the full neighboring Jobs matrix**

Run the Task 13 neighboring command from `Docs/superpowers/plans/2026-07-14-jobs-admission-hardening-and-lease-lifecycle.md` without deselecting the two corrected cases. Expected: 104 passes and zero skips.

Execution evidence: all 104 neighboring tests passed with the corrected cases included, required real PostgreSQL, zero skips, and zero deselections.

- [x] **Step 3: Run mechanical and security checks**

Run:

```bash
source ../../.venv/bin/activate
python -m ruff check tldw_Server_API/tests/Jobs/test_jobs_dependency_acquire_counter_regressions.py
python -m compileall -q tldw_Server_API/tests/Jobs/test_jobs_dependency_acquire_counter_regressions.py
python -m bandit -r tldw_Server_API/tests/Jobs/test_jobs_dependency_acquire_counter_regressions.py \
  -f json -o /tmp/bandit_task_12988.json
git diff --check
```

Expected: Ruff, compile, and diff checks pass; Bandit reports no unexpected findings, with test assertions classified as expected test-only `B101` findings if present.

Execution evidence: Ruff, compileall, and `git diff --check` pass. The full Bandit report contains 110 expected test-only `B101` assertion findings and zero errors; the follow-up scan excluding `B101` contains zero findings and zero errors.

- [x] **Step 4: Close merged lifecycle tracking**

Mark Stage 5 and Task 13 Step 4 complete in the prior plan. Update `TASK-12969.3` with PR #2763 merge commit `616d6dd35d48849f22b320d34823bfcfecbc4b74`, final verification, the merged requester-summary policy deviation, and a final summary; then mark it Done.

- [x] **Step 5: Preserve the ambiguous parent records**

Record in `TASK-12988` that four files currently claim `TASK-12969`, making parent mutation unsafe through Backlog MCP. Do not manually edit any parent record.

Execution evidence: `TASK-12969.3` is Done with merge and verification evidence; Stage 5 and Task 13 Step 4 are complete. The four ambiguous `TASK-12969` parent records remain untouched and the collision is recorded in both child and stabilization tasks.

- [x] **Step 6: Commit and open the stabilization PR**

Stage only the test, plan, and Backlog files owned by this task. Commit with a message explaining that the change reconciles the test baseline with already-merged durability semantics, then open a PR against `dev` with a requester-owned Change summary placeholder.

Execution evidence: commit `f9a5b8e733` was pushed on `codex/jobs-admission-baseline-reconcile`, and draft PR #2765 was opened against `dev`: https://github.com/rmusser01/tldw_server/pull/2765. The PR remains merge-blocked until the requester replaces the Change summary placeholder with their own explanation of what changed and why.
