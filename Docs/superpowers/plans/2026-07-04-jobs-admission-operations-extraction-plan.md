# Jobs Admission Operations Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the Jobs create/admission transaction from `JobManager.create_job` into backend-specific operation modules while preserving the public facade and current behavior.

**Architecture:** `JobManager` keeps caller validation, payload hygiene, allowed-queue/job-type policy, fair-share priority adjustment, public row mapping, metrics, audit, and in-process event fanout. SQLite and Postgres admission modules own transactional quota checks, idempotent insert/select, counter updates, and transactional `job.created` rows. Operation modules return typed `AdmissionResult` facts and must not import `JobManager`.

**Tech Stack:** Python dataclasses, sqlite3, psycopg-compatible cursor usage, existing Jobs migrations, pytest, FastAPI TestClient only for existing API contracts, Bandit.

---

## Scope

This is `TASK-12138`. It implements rollout step 7 from `Docs/superpowers/specs/2026-06-24-jobs-backend-parity-refactor-design.md`.

In scope:
- `create_job` admission only.
- SQLite and Postgres operation modules.
- Idempotent create and current-request `job.created` event behavior.
- Create-time quota checks, counters, durable event rows, and facade-owned side effects.
- Focused parity, fault-injection, and contract verification.

Out of scope:
- `acquire_next_job`, `renew_job_lease`, `complete_job`, `fail_job`, `cancel_job`, batch lifecycle, dependencies, prune/archive, read-model/admin SQL extraction.
- Public REST response changes.
- Schema changes.

## Existing Baseline

Baseline command run before planning:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_completion_idempotent_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_queue_controls_and_admin_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_events_outbox_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_settings.py \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py \
  tldw_Server_API/tests/Jobs/test_jobs_admin_contract_sqlite.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py \
  -q
```

Result: `55 passed, 424 warnings`.

## File Structure

- Modify: `tldw_Server_API/app/core/Jobs/operations/contracts.py`
  - Permit idempotent existing admission results to carry durable event facts because current create replay writes a transactional `job.created` row.
  - Allow `CreateJobCommand.project_id` to accept the existing manager input shape.
- Create: `tldw_Server_API/app/core/Jobs/operations/sqlite/__init__.py`
  - Export SQLite admission helper.
- Create: `tldw_Server_API/app/core/Jobs/operations/sqlite/admission.py`
  - SQLite transactional admission implementation.
- Create: `tldw_Server_API/app/core/Jobs/operations/postgres/__init__.py`
  - Export Postgres admission helper.
- Create: `tldw_Server_API/app/core/Jobs/operations/postgres/admission.py`
  - Postgres transactional admission implementation.
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
  - Build `CreateJobCommand`, call backend operation helper, map result to existing public row, and emit facade-owned side effects.
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py`
  - Add red coverage for idempotent existing durable event facts and operation-package no-`JobManager` import coverage.
- Create: `tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py`
  - Direct SQLite operation tests for inserted, idempotent existing, quota rejection, counters, and rollback on `job_events` failure.
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_manager_pg_fakes_extended.py`
  - Keep fake cursor compatibility and add a manager-routing assertion that Postgres create still uses current request context for idempotent replay events after extraction.
- Modify: `tldw_Server_API/tests/Jobs/parity/scenarios.py`
  - Add one shared scenario asserting idempotent replay persists a second transactional `job.created` event with current request/trace ids while preserving the original job row request/trace ids.
- Modify: `tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py`
  - Run the new shared scenario against SQLite.
- Modify: `tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py`
  - Run the same scenario against real Postgres when available.
- Modify: `backlog/tasks/task-12138 - Extract-Jobs-admission-operations-behind-JobManager.md`
  - Record plan, verification, and final summary.

## Task 1: Preserve Contract Truth For Idempotent Replay Events

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/operations/contracts.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py`

- [x] **Step 1: Write the failing contract test**

Add a test named `test_admission_existing_can_report_idempotent_durable_event` that constructs:

```python
result = AdmissionResult.existing(
    row={"id": 1, "status": "queued"},
    durable_events=({"event_type": "job.created", "attrs": {"idempotent": True}},),
)
```

Expected assertions:
- `result.outcome is OperationOutcome.NO_TRANSITION`
- `result.no_transition_reason is NoTransitionReason.IDEMPOTENT_EXISTING`
- `result.durable_events` contains the deep-copied event.

- [x] **Step 2: Verify red**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py::test_admission_existing_can_report_idempotent_durable_event -q
```

Expected: fails because `AdmissionResult.existing` does not accept `durable_events` and/or current invariants reject durable events on no-transition results.

- [x] **Step 3: Update the contract minimally**

Change `AdmissionResult.__post_init__` so durable events are allowed for:
- `OperationOutcome.APPLIED`
- `OperationOutcome.NO_TRANSITION` with `NoTransitionReason.IDEMPOTENT_EXISTING`

Change `AdmissionResult.existing` to accept `durable_events: Sequence[dict[str, Any]] = ()` and pass them through.

Do not relax `LifecycleResult`; lifecycle no-transition durable events remain invalid in this slice.

- [x] **Step 4: Verify green**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py -q
```

Expected: all operation contract tests pass.

## Task 2: Add Direct SQLite Admission Operation Tests

**Files:**
- Create: `tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py`
- Create later in Task 3: `tldw_Server_API/app/core/Jobs/operations/sqlite/admission.py`

- [x] **Step 1: Write failing module import and inserted-row test**

Create tests that import:

```python
from tldw_Server_API.app.core.Jobs.operations.contracts import CreateJobCommand, OperationOutcome
from tldw_Server_API.app.core.Jobs.operations.sqlite.admission import create_job_admission
```

The first test should:
- create a temp SQLite jobs DB with `ensure_jobs_tables`
- open a connection with row factory
- call `create_job_admission` with a `CreateJobCommand(domain="admission", queue="default", job_type="insert", payload={"x": 1}, owner_user_id="u1", request_id="req-1", trace_id="trace-1")`
- pass `uuid_value="uuid-insert"`, `now=datetime(2026, 1, 1, tzinfo=timezone.utc)`, `max_queued_quota=0`, `submits_per_minute_quota=0`, and `counters_enabled=True`
- assert `result.outcome is OperationOutcome.APPLIED`
- assert `result.inserted is True`
- assert `result.row["status"] == "queued"`
- assert exactly one `job.created` row exists with matching `request_id` and `trace_id`
- assert counters incremented ready count once.

- [x] **Step 2: Write failing idempotent-existing test**

The second test should:
- insert once with idempotency key `same`
- replay with the same idempotency key and new request/trace ids
- assert the replay returns `OperationOutcome.NO_TRANSITION`
- assert `result.inserted is False`
- assert the returned row keeps the original row request/trace ids
- assert two transactional `job.created` rows exist
- assert the second event uses the replay request/trace ids.

- [x] **Step 3: Write failing quota and rollback tests**

Add quota tests for:
- max queued quota rejection returns `OperationOutcome.ADMISSION_REJECTED` with `AdmissionRejectionReason.QUOTA_EXCEEDED`
- submits-per-minute quota rejection returns the same rejection reason.

Add rollback test with a connection wrapper that raises `sqlite3.OperationalError` on `INSERT INTO job_events`; assert no `jobs` row commits after the exception.

- [x] **Step 4: Verify red**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py -q
```

Expected: collection fails with `ModuleNotFoundError` for the new SQLite admission module.

## Task 3: Implement SQLite Admission Operation

**Files:**
- Create: `tldw_Server_API/app/core/Jobs/operations/sqlite/__init__.py`
- Create: `tldw_Server_API/app/core/Jobs/operations/sqlite/admission.py`

- [x] **Step 1: Add package marker**

Create `sqlite/__init__.py` with:

```python
"""SQLite Jobs admission operations."""

from .admission import create_job_admission

__all__ = ["create_job_admission"]
```

- [x] **Step 2: Implement `create_job_admission`**

Signature:

```python
def create_job_admission(
    conn: sqlite3.Connection,
    *,
    command: CreateJobCommand,
    uuid_value: str,
    now: datetime,
    max_queued_quota: int,
    submits_per_minute_quota: int,
    counters_enabled: bool,
) -> AdmissionResult:
```

Rules:
- Use `with conn:` for the transaction.
- Serialize `command.payload` with `json.dumps(command.payload)`.
- Normalize aware `command.available_at` to UTC naive string for SQLite.
- Perform the existing max queued and submits-per-minute checks.
- Return `AdmissionResult.rejected(AdmissionRejectionReason.QUOTA_EXCEEDED, message=...)` for quota rejections instead of raising.
- For idempotent inserts, use the existing `INSERT OR IGNORE` plus `SELECT` behavior.
- For idempotent replay, insert a `job.created` row with `attrs_json.idempotent=true`, current `command.request_id`, and current `command.trace_id`.
- For inserted rows, insert a `job.created` row with `attrs_json.idempotent=false`.
- Update `job_counters` only for inserted rows when `counters_enabled` is true.
- Return `AdmissionResult.applied(...)` for inserted rows and `AdmissionResult.existing(..., durable_events=(event_fact,))` for idempotent existing rows.
- Let `sqlite3.OperationalError` from `job_events` insert propagate so the transaction rolls back.

- [x] **Step 3: Verify green**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py -q
```

Expected: all direct SQLite admission operation tests pass.

## Task 4: Route SQLite `JobManager.create_job` Through The Operation

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`

- [x] **Step 1: Write/extend facade regression tests first**

Use existing tests plus add a shared parity scenario in Task 6 before changing behavior. The manager must still:
- raise `ValueError` for quota rejection
- preserve original request/trace ids on idempotent row replay
- write a replay `job.created` event with current request/trace ids
- not increment created metrics when the transactional event insert fails
- use exactly one facade event/audit path.

- [x] **Step 2: Add helper methods in `JobManager`**

Add small private helpers:
- `_build_create_job_command(...) -> CreateJobCommand`
- `_map_admission_result(result: AdmissionResult) -> dict[str, Any]`
- `_emit_create_side_effects(result: AdmissionResult, *, backend: str) -> None`

Keep helpers in `manager.py` for this slice. Do not move secret scanning, encryption, queue allowlist, job-type allowlist, fair-share, metrics, audit, or in-process event fanout into operation modules.

- [x] **Step 3: Replace SQLite inline SQL branch**

In the SQLite branch of `create_job`:
- call `sqlite_create_job_admission(...)`
- pass quota values from `_quota_get`
- pass `counters_enabled=JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", ""))`
- call `_update_gauges` after success, matching existing SQLite behavior
- convert `AdmissionResult.rejected(...QUOTA_EXCEEDED...)` to the same `ValueError` messages as today
- call side effects after the transaction returns.

- [x] **Step 4: Verify SQLite facade behavior**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_fault_injection_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_events_sqlite.py \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  -q
```

Expected: all selected SQLite facade/fault/parity tests pass.

## Task 5: Implement And Route Postgres Admission Operation

**Files:**
- Create: `tldw_Server_API/app/core/Jobs/operations/postgres/__init__.py`
- Create: `tldw_Server_API/app/core/Jobs/operations/postgres/admission.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_manager_pg_fakes_extended.py`

- [x] **Step 1: Add package marker**

Create `postgres/__init__.py` with:

```python
"""Postgres Jobs admission operations."""

from .admission import create_job_admission

__all__ = ["create_job_admission"]
```

- [x] **Step 2: Implement `create_job_admission`**

Signature:

```python
def create_job_admission(
    conn: Any,
    cursor_factory: Callable[[Any], ContextManager[Any]],
    *,
    command: CreateJobCommand,
    uuid_value: str,
    now: datetime,
    max_queued_quota: int,
    submits_per_minute_quota: int,
    counters_enabled: bool,
) -> AdmissionResult:
```

Rules:
- Use `with conn:` and `with cursor_factory(conn) as cur:`.
- Keep SQL text and parameter ordering compatible with existing fake cursor tests.
- Use `ON CONFLICT (domain, queue, job_type, idempotency_key) DO NOTHING RETURNING *` for idempotent insert.
- Select existing row on conflict.
- Insert transactional `job.created` event for inserted and idempotent-existing cases, using current request/trace ids.
- Update counters only when inserted and `counters_enabled` is true.
- Return the same `AdmissionResult` shapes as SQLite.
- Let event insert errors propagate so the transaction fails.

- [x] **Step 3: Route the Postgres branch in `JobManager.create_job`**

Replace only the Postgres create/admission SQL block with a call to `postgres_create_job_admission(...)`. Keep the facade side-effect helper shared with SQLite.

- [x] **Step 4: Verify fake Postgres behavior**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_manager_pg_fakes_extended.py -q
```

Expected: all fake Postgres manager tests pass.

## Task 6: Add Shared Parity Coverage For Idempotent Replay Events

**Files:**
- Modify: `tldw_Server_API/tests/Jobs/parity/scenarios.py`
- Modify: `tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py`
- Modify: `tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py`

- [x] **Step 1: Add shared scenario**

Add `run_idempotent_replay_records_current_request_event_scenario(make_manager)`:
- create a job with idempotency key `event-replay-key`, request id `request-first`, trace id `trace-first`
- replay with request id `request-second`, trace id `trace-second`
- assert both calls return the same job id
- assert replay row keeps `request-first` and `trace-first`
- query `list_job_events_after(after_id=0, domain="parity", queue="default", job_type="event-replay", event_types=("job.created",), limit=20)`
- assert two `job.created` rows exist for the job
- assert the last event uses `request-second` and `trace-second`.

- [x] **Step 2: Wire SQLite and Postgres wrappers**

Add wrapper tests:
- `test_sqlite_idempotent_replay_records_current_request_event`
- `test_postgres_idempotent_replay_records_current_request_event`

- [x] **Step 3: Verify parity wrappers**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py \
  -q -rs
```

Expected: SQLite passes. Postgres passes when the fixture is reachable or skips with the existing explicit Postgres-unreachable reason.

## Task 7: Full Focused Verification, Backlog, And Commit

**Files:**
- Modify: `backlog/tasks/task-12138 - Extract-Jobs-admission-operations-behind-JobManager.md`

- [x] **Step 1: Run focused Jobs admission matrix**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py \
  tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_quotas_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_fault_injection_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_manager_pg_fakes_extended.py \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py \
  tldw_Server_API/tests/Jobs/test_jobs_admin_contract_sqlite.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py \
  -q -rs
```

Expected: all SQLite/fake/unit/API contract tests pass; Postgres tests pass or record existing fixture skip.

- [x] **Step 2: Run import-boundary scan**

Run:

```bash
rg -n "JobManager|Jobs\\.manager|from .*manager" tldw_Server_API/app/core/Jobs/operations
```

Expected: no matches.

- [x] **Step 3: Run Bandit**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m bandit -q -s B101 -r \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/operations \
  tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py
```

Expected: exits 0.

- [x] **Step 4: Run diff hygiene and compile smoke**

Run:

```bash
git diff --check
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m py_compile \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/operations/contracts.py \
  tldw_Server_API/app/core/Jobs/operations/sqlite/admission.py \
  tldw_Server_API/app/core/Jobs/operations/postgres/admission.py
```

Expected: both commands exit 0.

- [x] **Step 5: Update Backlog task**

Record:
- plan path
- focused baseline result
- red/green test evidence
- final verification results
- Postgres fixture skips, if any
- Bandit result
- final summary.

- [ ] **Step 6: Commit**

Run:

```bash
git add \
  Docs/superpowers/plans/2026-07-04-jobs-admission-operations-extraction-plan.md \
  "backlog/tasks/task-12138 - Extract-Jobs-admission-operations-behind-JobManager.md" \
  tldw_Server_API/app/core/Jobs/manager.py \
  tldw_Server_API/app/core/Jobs/operations/contracts.py \
  tldw_Server_API/app/core/Jobs/operations/sqlite \
  tldw_Server_API/app/core/Jobs/operations/postgres \
  tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py \
  tldw_Server_API/tests/Jobs/parity/scenarios.py \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py \
  tldw_Server_API/tests/Jobs/test_jobs_manager_pg_fakes_extended.py
git commit -m "refactor(jobs): extract admission operations"
```

Expected: commit succeeds.

## Review Checklist

- `JobManager.create_job` remains the only public caller entrypoint.
- Operation modules contain backend SQL and do not import `JobManager`.
- Queue allowlist, fair-share, secret hygiene, encryption, job-type allowlist, and side effects stay facade-owned.
- Idempotent replay still writes a transactional `job.created` event with current request/trace ids.
- Existing public row request/trace ids are preserved on idempotent replay.
- Metrics increment only for inserted rows and only after transactional admission succeeds.
- Event insert failure rolls back job insertion and does not increment created metrics.
- No lifecycle methods are extracted.
