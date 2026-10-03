# Claims Jobs Stage 2B Review Metrics Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development to implement the tasks with specification and code-quality reviews. Steps use checkbox syntax for tracking.

**Goal:** Run recurring Claims review aggregation through owner-scoped shared Jobs while preserving the opt-in local fallback and repairing the validated persistence, scope, discovery, and scheduler issues.

**Architecture:** A thin APScheduler callback captures one UTC execution window and admits one Job per eligible owner. Claims validates payloads and owns aggregation; package-owned Media DB operations own SQL and atomic writes. Jobs owns leases, retries, cancellations, and administration.

**Tech Stack:** Python, APScheduler, WorkerSDK, MediaDatabase, SQLite, PostgreSQL, pytest, Hypothesis, Ruff, Bandit.

**Backlog:** TASK-9935.3
**Specification:** `Docs/superpowers/specs/2026-09-07-claims-jobs-stage2b-review-metrics-design.md`
**Base:** `9958110df2a9011e19f48b0eae821353e19d4af8` (`origin/dev`, fetched 2026-10-02)
**Environment:** Activate `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate` before Python commands. Work only in this isolated worktree. Use the official PostgreSQL fixtures; document genuine fixture unavailability.

## Stage 1: Existing-Only Database Sessions

**Goal:** Add a shared opt-in session mode that cannot create or initialize missing SQLite data.
**Success Criteria:** Normal callers retain creating behavior; existing-only callers use a private `mode=rw` backend, create no directories, and skip schema bootstrap.
**Tests:** Missing path, existing data, deletion at open, permissions, registry isolation, PostgreSQL routing.
**Status:** Not Started

### Task 1: Shared Media DB Open Contract

**Files:**
- Modify `tldw_Server_API/app/core/DB_Management/media_db/api.py`
- Modify `tldw_Server_API/app/core/DB_Management/media_db/runtime/factory.py`
- Modify `tldw_Server_API/app/core/DB_Management/media_db/runtime/bootstrap_lifecycle_ops.py`
- Modify `tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py` only if necessary to suppress parent creation for `mode=rw`
- Create `tldw_Server_API/tests/DB_Management/test_media_db_existing_only.py`

- [ ] Add a failing behavioral test using a real temporary SQLite file:

```python
def test_existing_only_does_not_create_missing_database(tmp_path):
    path = tmp_path / "absent" / "media.db"
    with pytest.raises(FileNotFoundError):
        with managed_media_database("42", db_path=str(path), existing_only=True):
            pass
    assert not path.parent.exists()
```

- [ ] Run `python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_existing_only.py -q` and record the missing-argument failure.
- [ ] Add `existing_only: bool = False` to the public create/managed factories, runtime factory, and constructor. In this mode bypass parent creation and `_initialize_schema`, and bypass the managed `initialize_db` call. Route SQLite through a dedicated `SQLiteBackend(DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=path.as_uri() + "?mode=rw"))`. Do not reuse the path-keyed creating registry backend. Eagerly verify the connection through the shared layer so an absent file reports `FileNotFoundError`, while permission/ambiguous open errors retain their original failure. Close the dedicated backend pool at session exit.
- [ ] Add tests that deleting the file after any precheck cannot recreate it, a permission/open failure cannot be a successful absence skip, existing rows remain readable/writable, and creating callers are unchanged.
- [ ] Run the new tests and relevant shared backend/factory/session tests; run scoped Ruff, compilation, and Bandit. Parent records results and commits the scoped patch.

## Stage 2: Atomic Owner-Window Aggregation

**Goal:** Move domain aggregation out of claims_service, centralize its SQL, and make one window atomic and consistent.
**Success Criteria:** UTC bounds/grouping, stable JSON, serialized owner calculations, no per-row commit, one source snapshot, bounded discovery.
**Tests:** Counts/reasons, UTC boundaries, multi-group rollback, duplicate upserts, owner isolation and RLS, lookback bounds.
**Status:** Not Started

### Task 2: Media DB Operations And Domain Adapter

**Files:**
- Modify `tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_review_metrics_ops.py`
- Modify helper imports/bindings in `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py`
- Create `tldw_Server_API/app/core/Claims_Extraction/claims_review_metrics.py`
- Modify only `aggregate_claims_review_extractor_metrics_daily` in `tldw_Server_API/app/core/Claims_Extraction/claims_service.py`
- Modify `tldw_Server_API/tests/DB_Management/test_media_db_claims_review_metrics_ops.py`
- Create `tldw_Server_API/tests/Claims/test_claims_review_metrics_window.py`
- Create `tldw_Server_API/tests/MediaDB2/test_claims_review_metrics_postgres.py` using official fixtures

- [ ] Seed two owner databases and real review events; add failing tests for explicit windows and rollback after the second group write.

```python
def test_explicit_window_aggregates_seeded_review_events(review_db):
    written = aggregate_claims_review_metrics_window(
        db=review_db, owner_user_id="42",
        start_date=date(2026, 9, 6), end_date=date(2026, 9, 7),
    )
    row = review_db.list_claims_review_extractor_metrics_daily(user_id="42")[0]
    assert written == 1
    assert row["total_reviewed"] == 2
    assert json.loads(row["reason_code_counts_json"]) == {"spam": 1}
```

- [ ] Run the new tests red before implementing. Define these APIs:

```python
def aggregate_claims_review_metrics_window(*, db, owner_user_id: str,
                                         start_date: date, end_date: date) -> int: ...
def get_claims_review_metrics_window_rows(self, *, owner_user_id: str,
                                        start_date: date, end_date: date) -> list: ...
def list_claims_review_user_ids_page(self, *, start_date: date, end_date: date,
                                   after_user_id: str | None = None,
                                   limit: int = 100) -> list[str]: ...
```

- [ ] Validate dates and canonical owner at the domain boundary; reject reversed/over-366-day windows. Use one `db.transaction()` for the complete window, acquiring PostgreSQL `pg_advisory_xact_lock(hashtextextended(?, 0))` with `claims-review-metrics:{owner}` before the source query. SQLite uses existing `BEGIN IMMEDIATE`.
- [ ] Build one shared filtered source CTE in the DB module. SQLite day is `DATE(l.created_at)`; PostgreSQL day is `(l.created_at AT TIME ZONE 'UTC')::date`, parameters are aware UTC datetimes. The CTE projects day/extractor/version/new_status/old_text/new_text/reason_code. `UNION ALL` one metrics group projection and one reason group projection, tagged by `kind`, with matching column types. Use counts and conditional sums matching the existing statuses/edit rules. This gives one statement snapshot without a pre-lock repeatable-read snapshot.
- [ ] Replace select/insert/update with `INSERT ... ON CONFLICT(user_id,report_date,extractor,extractor_version) DO UPDATE` assigning all count/JSON fields and `updated_at`; preserve `created_at` and row return. Join caller transaction or create one for standalone upserts; never commit per row inside an outer transaction.
- [ ] Assemble reason maps and write stable `json.dumps(counts, sort_keys=True, separators=(",", ":"))`. Do not delete absent historical groups. Return groups written; no source rows returns zero. Keep the legacy function signature as a thin report_date/lookback adapter.
- [ ] Implement date-filtered PostgreSQL owner pages with `DISTINCT COALESCE(CAST(m.owner_user_id AS TEXT),m.client_id)`, strict `> after_user_id`, deterministic text ordering, and `LIMIT` clamped to 100. Retain the legacy all-owner helper for existing callers.
- [ ] Add real SQLite rollback/duplicate tests and official PostgreSQL tests for UTC parity, concurrent events, forced RLS, team/org/deleted owner media, distinct owner locks, and scope cleanup. Run focused tests and quality gates. Parent reviews and commits.

## Stage 3: Strict Jobs Contracts And Worker Handler

**Goal:** Register the new domain job in existing Claims Jobs integration.
**Success Criteria:** Exact payload/owner validation, typed admission, owner-scoped data handling, safe outcomes/errors, export classifier parity.
**Tests:** Strict types/dates, admission replay, owner mismatch, missing data, delayed execution, classifier matrix.
**Status:** Not Started

### Task 3: Contract, Producer Helper, And Handler

**Files:**
- Modify `tldw_Server_API/app/core/Claims_Extraction/claims_job_contracts.py`
- Modify `tldw_Server_API/app/core/Claims_Extraction/claims_jobs.py`
- Modify `tldw_Server_API/app/core/Claims_Extraction/claims_job_handlers.py`
- Modify `tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py`
- Modify `tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py`
- Modify `tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py`

- [ ] Add failing tests for the exact five-key payload, bool version/owner rejection, UTC-Z seconds, strict dates, 366-day cap, row owner equality and unknown keys.

```python
payload = {"version": 1, "owner_user_id": "42",
           "scheduled_for": "2026-09-07T00:00:00Z",
           "start_date": "2026-09-06", "end_date": "2026-09-07"}
assert validate_review_metrics_payload(payload) == payload
```

- [ ] Define `CLAIMS_AGGREGATE_REVIEW_METRICS_JOB_TYPE = "claims_aggregate_review_metrics"` and `validate_review_metrics_payload`. Reuse canonical owner validation; require `type(version) is int`. Round-trip parsed datetime/date through their canonical formats. Reject max date if an exclusive end cannot be represented.
- [ ] Add `claims_review_metrics_jobs_enabled(settings_obj=None)` and `enqueue_claims_review_metrics(*, owner_user_id, scheduled_for, start_date, end_date, interval_seconds, job_manager=None, settings_obj=None) -> AdmissionResult`. Use `admit_job`, existing queue/domain, priority 5, `_max_retries("CLAIMS_JOBS_MAX_RETRIES_REVIEW_METRICS", 3, settings_obj)`, and the exact spec key/batch group. Return typed result without inspecting timestamps or creating Claims queue controls.
- [ ] Generalize `_is_transient_export_storage_error` to `_is_transient_claims_storage_error`, retaining the old callable as an alias if tests/callers require it. Add a table for every existing supported SQLite code/message, SQLSTATE, errno, cause wrapping, connection/timeout and permanent exception; export behavior must remain identical.
- [ ] Handler validates row/payload owner before routing, runs a synchronous owner-window function via `asyncio.to_thread`, and creates/uses/closes Media DB inside `scoped_context(user_id=int(owner), is_admin=True)` with empty org/team lists. PostgreSQL keeps the explicit owner predicate. SQLite passes `existing_only=True`. The handler must not read current scheduler dates or add cancellation polling.
- [ ] Return `ok` or `skipped` with `start_date`, `end_date`, `groups_written`, and `reason` only for a skip. Verified absent SQLite data returns `owner_database_missing`; zero aggregates returns `no_activity`. Translate other exceptions into safe structured `ClaimsJobError`, retrying only explicit transient signals.
- [ ] Run the three Claims Jobs test modules plus analytics export handler regressions and scoped quality checks. Parent reviews and commits.

## Stage 4: Scheduler Integration And Verification

**Goal:** Replace the sleep loop with the configured APScheduler producer and retain a mutually exclusive local route.
**Success Criteria:** Immutable routing, bounded nonblocking fan-out, idempotency, safe configuration/shutdown and documented rollout.
**Tests:** Full flag matrix, overflow, midnight/leap properties, idempotent fan-out, transient-only retry, shutdown, startup lifecycle, API regressions.
**Status:** Not Started

### Task 4: APScheduler Producer

**Files:**
- Modify `tldw_Server_API/app/services/claims_review_metrics_scheduler.py`
- Modify its lifecycle provider/registration only where required by existing startup tests
- Modify `tldw_Server_API/tests/Claims/test_claims_review_metrics_scheduler.py`
- Modify `tldw_Server_API/tests/Services/test_claims_review_metrics_scheduler.py`
- Create `tldw_Server_API/tests/Claims/test_claims_review_metrics_jobs_scheduler.py`
- Modify `Docs/Operations/Env_Vars.md`
- Modify `Docs/Product/Claims_Module/Claims_Monitoring_Implementation.md`

- [ ] Add failing routing/normalization tests, including environment override precedence and false string values. Add a property test:

```python
@given(st.datetimes(timezones=st.just(timezone.utc),
                    min_value=datetime(2000, 1, 1), max_value=datetime(2100, 1, 1)),
       st.integers(min_value=60, max_value=86400 * 365),
       st.integers(min_value=1, max_value=366))
def test_window_identity_invariants(now, interval, lookback):
    slot = int(now.timestamp()) // interval * interval
    assert slot <= now.timestamp() < slot + interval
    assert (now.date() - timedelta(days=lookback - 1)).toordinal() <= now.date().toordinal()
```

- [ ] Resolve settings at startup into a frozen config: env overrides config, strict truth parsing, invalid/nonpositive interval 86400, floor positive interval to 60, invalid lookback 2, cap 366, metrics Jobs route requires both global and metrics flags. Worker-enabled flag does not route production.
- [ ] Construct UTC `IntervalTrigger` with `start_date=now+timedelta(seconds=min(5,interval))`; verify first and next fires before startup. Reject overflow and disable only this scheduler. Register `max_instances=1`, `coalesce=True`, `misfire_grace_time=None`.
- [ ] Preserve a task named `claims_review_metrics_scheduler` for existing lifecycle compatibility. Its wrapper owns an `asyncio.Event`, starts the scheduler, and on cancellation sets the stop event, stops callbacks, and awaits bounded callback cleanup. Accepted Jobs remain durable.
- [ ] One callback captures UTC now once. Use integer floor slot and explicit inclusive UTC dates. SQLite directory scan runs in a worker thread; PostgreSQL pages each open/use/close a fresh session in privileged maintenance scope. Fixed-user fallback only in single-user mode. Canonical validation skips bad owners.
- [ ] Each owner admission runs in a worker thread and reuses captured settings/key across at most three attempts. Only transient storage errors or typed backend conflicts retry at 0.25 and 1.0 seconds. Terminal rejection counts failed; only APPLIED counts accepted; only idempotent NO_TRANSITION counts deduplicated. No local call follows an admission attempt. Event-aware delays stop cooperatively.
- [ ] Local route calls the explicit-window aggregator per owner using the same DB routing/scope boundaries. Preserve the callable signature of `run_claims_review_metrics_once` for current injected tests/callers. Do not catch cancellation as an ordinary failure.
- [ ] Add behavior tests for paginated owner fan-out, one-owner failures, response-loss replay, thread offloading, shutdown while retrying, callback slot crossing midnight, and coalesced repair after >300 seconds. Preserve existing lifecycle/startup and metrics API tests.
- [ ] Document flags/defaults/bounds, producer-only deployments, best-effort cancellation, cutover and queue-change barriers, rollback, SQLite lock contention, scoped log guarantees, and TASK-9935.2 removal.

### Task 5: Integration, Reviews, And Closeout

- [ ] Run Claims Jobs contracts/enqueue/handlers, review metrics API/window/scheduler, Services worker/scheduler, existing-only/shared backend and DB operations tests together.
- [ ] Exercise admission -> WorkerSDK -> Media DB with real SQLite and official PostgreSQL fixtures; verify duplicate execution converges and cancelled queued work never runs. Record fixture skips only when infrastructure is unavailable.
- [ ] Request independent spec compliance review, address validated gaps, then request code quality review and address validated findings.
- [ ] Run scoped Ruff, compilation, whitespace and Bandit on changed production paths. Fix new findings. Record all evidence in TASK-9935.3.
- [ ] Commit scoped working units with Backlog references. Mark stage/task progress as each completes. Remove only this plan after every stage is verified complete, following repository guidance. Do not create or merge a PR without a current user request.
