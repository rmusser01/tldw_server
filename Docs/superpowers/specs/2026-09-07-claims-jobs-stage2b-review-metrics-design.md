# Claims Jobs Stage 2B Review Metrics Aggregation Design

**Status:** Approved for implementation planning
**Date:** 2026-09-07
**Revised:** 2026-10-02 after review against current `dev`
**Backlog:** TASK-9935.1
**Parent design:** `2026-06-24-claims-jobs-operational-control-plane-design.md`

## Objective

Move scheduled Claims review-metrics aggregation onto the shared Jobs control
plane without changing the public metrics API or the established interval-based,
rolling-lookback behavior.

Claims continues to own owner discovery, aggregation rules, database routing,
daily metric records, and domain outcomes. APScheduler owns only recurring
enqueue decisions. Jobs exclusively owns durable admission, leases, execution
status, retries, cancellation state, quarantine, and administrative controls.

The migration is opt-in and retains the current bounded local execution path for
one release. The local and Jobs paths are mutually exclusive for each scheduler
lifecycle.

## Current Behavior

`app/services/claims_review_metrics_scheduler.py` currently starts an in-process
`asyncio` sleep loop. After a short startup delay, each interval directly:

1. Enumerates SQLite owner database directories or PostgreSQL owners with review
   activity.
2. Opens the relevant Media DB.
3. Calls `aggregate_claims_review_extractor_metrics_daily` for each owner.
4. Repeats after `CLAIMS_REVIEW_METRICS_INTERVAL_SEC`.

The aggregation function lives in the oversized `claims_service.py`. It derives
dates at execution time, groups review-log events by date and the Claim row's
current extractor metadata, and persists rows through a select-then-insert/update
helper. Daily rows are unique by `(user_id, report_date, extractor,
extractor_version)`.

The current path has no durable execution record, owner-level retry isolation,
lease protection, cancellation state, or Jobs administration. Its date window
can also drift when work is delayed, PostgreSQL day grouping depends on session
timezone, and the persistence helper is vulnerable to concurrent insert races.

## Decisions

1. Preserve the existing interval and rolling-lookback configuration contract.
2. Use a thin APScheduler producer rather than event-driven aggregation or a
   coordinator Job.
3. Enqueue one owner-scoped Job per eligible owner and execution window, defined
   by `(slot_epoch, start_date, end_date)`.
4. Persist explicit UTC date bounds in every Job payload.
5. Keep all queue lifecycle and administration in the existing Jobs module.
6. Keep Claims as the source of truth for metric data and aggregation outcomes.
7. Retain a mutually exclusive compatibility-local path for one release.
8. Fix atomic-upsert and PostgreSQL UTC-grouping correctness as part of Stage 2B.
9. Do not add Claims queue-control endpoints or Claims-owned lease/retry state.

## Ownership Boundaries

### APScheduler Service

The Claims review-metrics scheduler owns:

- Resolving its immutable startup mode from configuration.
- Triggering an immediate catch-up callback without blocking application startup.
- Triggering subsequent callbacks at the configured interval.
- Capturing one UTC timestamp per callback and deriving its run slot and date
  window.
- Discovering eligible owner identifiers.
- Enqueuing one Job per owner with deterministic identity.
- Applying bounded, transient-only retries to Jobs admission.
- Logging aggregate producer outcomes.

It does not aggregate metrics, acquire Jobs work, manage leases, inspect worker
progress, retry domain execution, or expose queue controls.

### Jobs Module

Jobs owns:

- Durable Job admission and idempotency enforcement.
- Worker acquisition and lease renewal.
- Execution status, retry count, retry scheduling, cancellation state,
  quarantine, and terminal history.
- Existing owner-aware read and administration APIs.

No Claims-specific queue, lease table, retry loop, cancellation poller, or Jobs
administration endpoint is introduced.

### Claims Module

Claims owns:

- The review-metrics Job type and strict payload validation.
- Owner validation and owner-scoped Media DB resolution.
- Review-log aggregation and daily metric persistence.
- Safe domain result and error translation.
- The existing `/api/v1/claims/review/metrics` read behavior.

`claims_review_extractor_metrics_daily` remains the source of truth for completed
domain output. Jobs results are operational summaries, not replacement metric
records.

## Scheduling Contract

### Existing Controls

Stage 2B retains:

- `CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED`
- `CLAIMS_REVIEW_METRICS_INTERVAL_SEC`, default `86400`
- `CLAIMS_REVIEW_METRICS_LOOKBACK_DAYS`, default `2`

It adds:

- `CLAIMS_REVIEW_METRICS_JOBS_ENABLED`, default `false`
- `CLAIMS_JOBS_MAX_RETRIES_REVIEW_METRICS`, default `3`, constrained to the
  Jobs-supported `0..100` range

`CLAIMS_JOBS_ENABLED` remains the global Claims producer switch.
`CLAIMS_JOBS_WORKER_ENABLED` remains an independent local worker-startup switch
and never participates in producer routing.

### Configuration Normalization

The scheduler parses configuration once at startup and keeps the resulting mode
and values immutable until restart.

- Missing, non-numeric, or non-positive intervals use `86400` with a warning.
- Positive intervals below `60` seconds are raised to `60` with a warning.
- Intervals of `60` seconds or more are preserved; there is no upper clamp that
  could make work run more frequently than configured.
- Before registration, verify that APScheduler can represent the interval and
  calculate its first and next fire times. An out-of-range numeric interval
  disables this scheduler with a sanitized configuration warning; it is never
  replaced by a shorter interval. Recheck subsequent date overflow safely.
- Missing, non-numeric, or non-positive lookback values use `2` with a warning.
- Lookback is capped at `366` days with a warning to bound query and retry cost.

Configuration diagnostics report setting names and normalized values without
including raw exception text or sensitive environment content.

### APScheduler Behavior

The service uses an APScheduler interval trigger with:

- `max_instances=1`
- coalesced misfires
- unlimited misfire grace (`misfire_grace_time=None`), so an event-loop stall
  coalesces into a recent-window repair instead of discarding the callback
- explicit UTC scheduling

Startup registers one interval Job with its first run due after
`min(5, interval_seconds)` seconds and returns without owner discovery or
database work. This preserves the existing near-startup execution behavior while
keeping application readiness nonblocking. The same APScheduler Job handles
normal interval callbacks.

Each callback captures one timezone-aware `now_utc`. It calculates:

- `slot_epoch = floor(now_utc_epoch / interval_seconds) * interval_seconds`
- `scheduled_for` as the canonical UTC ISO representation of `slot_epoch`
- `end_date` as the UTC calendar date from the captured `now_utc`
- `start_date = end_date - (lookback_days - 1)`

All derivation uses the same captured timestamp. The explicit date range is also
part of Job identity. If a long interval bucket crosses UTC midnight, a restart
after midnight may intentionally enqueue a second catch-up Job for the new date
range instead of allowing one bucket's earlier Job to suppress it.

An extended scheduler outage is recovered only within the configured lookback
window. Stage 2B does not add a durable schedule ledger or arbitrary historical
backfill facility.

## Owner Discovery And Fan-Out

The producer creates one Job per canonical positive-integer owner identifier.

- SQLite discovers sorted per-user directories that contain the Media DB file.
  If none exist, it preserves the single-user fallback.
- PostgreSQL obtains distinct owner identifiers with Claims review-log activity
  in the captured UTC date window through a package-owned Media DB helper. Use
  deterministic keyset pagination with pages of at most 100 owners, rather than
  loading all historical owners into memory. A new owner appearing behind the
  cursor is repaired by the next rolling-window callback.
- The fixed-user fallback applies only in single-user authentication mode.
  An empty multi-user discovery result creates no Jobs.
- Invalid owner identifiers are skipped with sanitized diagnostics.
- Duplicate identifiers are removed before enqueue.
- Failure to enqueue one owner never stops later owners in the same callback.
- Complete discovery failure produces no Jobs and is retried by the next
  scheduler callback; it never triggers an alternate execution path.

The producer reports owners discovered, accepted, deduplicated, and failed. It
does not persist its own fan-out state. Shared Jobs idempotency makes concurrent
application schedulers and restarts converge on the same Jobs rows.

PostgreSQL discovery runs inside an explicit `scoped_context(user_id=None,
is_admin=True)` maintenance context because the owner join traverses forced Media
RLS. Aggregation also uses a narrowly bounded privileged maintenance context with
the validated owner and an explicit SQL owner predicate; this includes the
owner's team/org and soft-deleted media consistently with historical aggregation.
Never inherit request org/team membership or expose this scope through an API.
Create, use, and close each Media DB session inside that scope before restoring
the prior context, and test scope cleanup on failure and pooled reuse. Jobs RLS
uses its existing separate worker/admission controls.

Filesystem discovery, each database page, and each Jobs admission execute in a
worker thread. Async callbacks yield between pages and owners and check a
shutdown event before discovery, each admission, and each retry delay. This
keeps API readiness and the event loop responsive without a producer cursor
ledger or coordinator Job.

## Job Contract

### Type And Payload

The new job type is:

```text
claims_aggregate_review_metrics
```

The version 1 payload contains exactly:

```json
{
  "version": 1,
  "owner_user_id": "42",
  "scheduled_for": "2026-09-07T00:00:00Z",
  "start_date": "2026-09-06",
  "end_date": "2026-09-07"
}
```

Validation requires:

- An exact known key set and supported payload version.
- A canonical positive-integer owner string within the shared Claims owner bound.
- A canonical UTC `scheduled_for` value with second precision and a `Z` suffix.
- Strict `YYYY-MM-DD` dates.
- `start_date <= end_date`.
- An inclusive window no larger than `366` days.
- No database paths, review text, reason-code data, credentials, filters, or
  arbitrary metadata.

The Jobs row's `owner_user_id` must exactly match the normalized payload owner.
Owner mismatch is a non-retryable scope violation.

### Admission Metadata

The producer creates the Job with:

- domain `claims`
- the existing `CLAIMS_JOBS_QUEUE`
- owner equal to the validated payload owner
- priority `5`
- the configured review-metrics retry limit
- batch group `claims-review-metrics:{start_date}:{end_date}`

The exact idempotency key is:

```text
claims:review_metrics:v1:{owner_user_id}:{interval_seconds}:{slot_epoch}:{start_date}:{end_date}
```

Admission metadata is derived internally and is never accepted from an external
caller.

The producer uses the typed `JobManager.admit_job` interface. An `APPLIED`
outcome counts as newly accepted; an idempotent `NO_TRANSITION` outcome counts as
deduplicated. Both are successful producer outcomes and return the authoritative
Jobs row. Rejected admission and backend conflict retain typed terminal or
transient handling instead of being inferred from row timestamps.

Jobs idempotency remains scoped by `(domain, queue, job_type, idempotency_key)`.
Changing `CLAIMS_JOBS_QUEUE` can therefore admit a second Job for the same domain
window. Atomic Claims persistence makes the resulting domain writes idempotent,
but rollout documentation must still treat queue changes as a transition.

### Admission Retry Boundary

The producer makes at most three total attempts for a recognized transient Jobs
admission failure. It waits `0.25` seconds before the second attempt and `1.0`
second before the third, always using the same idempotency key. Validation,
authorization, quota, and other terminal admission failures are not retried.

After any admission attempt, the producer never runs local aggregation for that
owner and slot. This avoids execution ambiguity if Jobs committed a row but the
producer lost the response.

## Worker Execution

The existing Claims Jobs worker dispatches the new type through WorkerSDK.

The handler:

1. Validates the payload and asserts Jobs-row/payload owner equality.
2. Opens the owner Media DB through `managed_media_database`. SQLite uses the
   new opt-in `existing_only=True` contract in the shared Media DB layer, backed
   by a dedicated `mode=rw` backend, with no directory creation or schema
   initialization. It must never reuse a creating backend from the registry.
3. Derives a SQLite path only for SQLite routing; PostgreSQL uses the configured
   shared backend plus explicit owner scoping.
4. Calls the Claims aggregation function in a worker thread with the payload's
   exact dates and owner.
5. Returns a non-sensitive domain result or raises a structured `ClaimsJobError`.

The handler never recalculates dates from current time, reads scheduler settings,
creates child Jobs, or implements its own lease and retry logic.

## Aggregation Refactor

The aggregation implementation moves from `claims_service.py` into
`claims_review_metrics.py`. That module exposes
`aggregate_claims_review_metrics_window`, whose dates and owner are always
explicit. `claims_service.aggregate_claims_review_extractor_metrics_daily`
remains as a thin compatibility adapter with its current signature for the local
fallback and existing callers; it resolves the legacy `report_date` or lookback
arguments and delegates to the explicit-window function.

`aggregate_claims_review_metrics_window` accepts explicit:

- Media DB instance
- owner identifier
- start date
- end date

It does not read scheduling configuration or default to the execution date.

The source query uses a half-open UTC interval:

```text
[start_date 00:00:00Z, end_date + 1 day 00:00:00Z)
```

SQLite retains its stored UTC timestamp representation. PostgreSQL uses
timezone-aware UTC parameters and groups with an explicit UTC expression rather
than the connection's session timezone. PostgreSQL applies the owner predicate;
SQLite relies on the per-owner database boundary while still writing the
canonical owner identifier into daily rows.

Reason-code count JSON is serialized with stable ordering and compact separators.
Jobs results contain only the number of metric groups written and the date range.

Move aggregate SQL into package-owned Media DB operations. Compute metric totals
and reason-code counts in one source statement (a shared filtered CTE plus
`UNION ALL` result kinds) so both observe one PostgreSQL statement snapshot at
READ COMMITTED. Do not assume two SELECTs in a transaction share a snapshot.

## Persistence Correctness

`aggregate_claims_review_metrics_window` performs its source reads and all daily
metric writes inside one Media DB transaction for the target owner and window.

- SQLite uses the Media DB's existing `BEGIN IMMEDIATE` transaction behavior.
- PostgreSQL first acquires a transaction-scoped advisory lock using
  `hashtextextended` over `claims-review-metrics:{owner_user_id}`.
- The lock is per owner, so unrelated PostgreSQL owners remain concurrent while
  overlapping windows for one owner are serialized.

The single source statement runs after this serialization boundary is acquired. This is
required because an atomic row upsert alone can still allow a calculation that
read older source data to overwrite a later calculation's newer counts.

Within that transaction,
`upsert_claims_review_extractor_metrics_daily` changes from a select followed by
insert or update to one backend-portable `INSERT ... ON CONFLICT ... DO UPDATE`
operation against the existing unique key. The helper uses the caller's active
transaction connection and does not commit per row.

This removes the concurrent initial-insert race, prevents stale overlapping
aggregations from finishing out of order, and makes duplicate worker execution
converge. The helper continues to return the persisted row and retains
SQLite/PostgreSQL API parity.

Stage 2B preserves current event-aggregation semantics: a rerun updates every
extractor group observed in its source window but does not delete historical
groups that are absent from a later query. Exact per-day replacement is not
introduced because review-log rows do not independently snapshot extractor
metadata; replacement would be a separate data-model and audit-semantics change.

If a source query or multi-group write fails, the transaction rolls back the
complete owner window. WorkerSDK retries only when the error is classified as
transient. The default two-day window, the 366-day hard cap, and the existing
review-log date index reduce query cost; they do not impose a runtime timeout.
Operational documentation warns that
large lookbacks increase SQLite writer contention.

## Results, Errors, And Cancellation

Successful execution returns:

```json
{
  "outcome": "ok",
  "start_date": "2026-09-06",
  "end_date": "2026-09-07",
  "groups_written": 4
}
```

No source activity returns a completed skipped outcome:

```json
{
  "outcome": "skipped",
  "reason": "no_activity",
  "start_date": "2026-09-06",
  "end_date": "2026-09-07",
  "groups_written": 0
}
```

An SQLite owner database removed between discovery and execution is skipped as
`owner_database_missing`. A PostgreSQL owner with no matching activity is
`no_activity`.

Only positively verified filesystem absence (`FileNotFoundError`/`ENOENT`) can
produce `owner_database_missing`; permission errors, malformed databases, and
ambiguous open failures remain failures under the storage classifier. An
existence precheck alone does not prevent a deletion/open race: the actual SQLite
connection must use `mode=rw`. Suppressing initialization errors is prohibited.

Retryable failures are limited to explicit temporary database, lock, connection,
timeout, and filesystem signals. Invalid payloads, owner mismatches, unsupported
versions, schema defects, query/programming errors, and unclassified exceptions
are non-retryable. Claims producer/handler diagnostics and persisted Jobs errors
never include raw exception text, paths, SQL, or database connection details.
Existing shared Media DB logs are outside this scoped logging guarantee; do not
promise global log sanitization without a separate shared-layer change.

The export-specific storage classifier in the current handler is
generalized into a shared Claims storage-failure classifier and reused by both
export and review-metrics handlers. This avoids divergent retry policy.
Preserve export behavior through a table-driven compatibility matrix covering
every current SQLite code/message, PostgreSQL SQLSTATE, OS errno, connection/
timeout signal, wrapped cause, and generic non-retryable exception.

Jobs excludes already-cancelled queued work from acquisition and owns the
terminal cancellation transition. The current Claims worker does not wire
WorkerSDK's optional pre-handler `cancel_check`, and WorkerSDK cannot interrupt
the synchronous database query after dispatch. A cancellation after acquisition
is therefore best effort: Jobs transitions the record to cancelled and rejects
the stale worker's late completion, while the idempotent Claims write may still
finish. Stage 2B does not add Claims-owned Jobs polling or a cancellation
mechanism inside the aggregation query.

## Rollout Matrix

At scheduler startup, mode is resolved once:

| Scheduler enabled | Global Claims Jobs | Metrics Jobs | Behavior |
| --- | --- | --- | --- |
| false | any | any | No recurring review-metrics work |
| true | true | true | Enqueue owner-scoped Jobs only |
| true | false | true | Compatibility-local mode plus configuration warning |
| true | any | false | Compatibility-local mode |

`CLAIMS_JOBS_WORKER_ENABLED=false` does not change this matrix. It supports
producer-only API processes and dedicated worker deployments; accepted Jobs stay
durable until an eligible worker is available.

The compatibility-local callback continues to invoke the existing bounded
per-owner aggregation path. A Jobs-mode callback never falls back locally after
owner discovery or Jobs admission failure.

### Enabling Jobs Mode

1. Deploy code and database changes with the new metrics Jobs flag disabled.
2. Enable the Claims Jobs worker in at least one worker process and verify queue
   health.
3. Enable `CLAIMS_JOBS_ENABLED` and `CLAIMS_REVIEW_METRICS_JOBS_ENABLED` on
   scheduler-producing processes.
4. Confirm owner-scoped Jobs and Claims metric outcomes before broad rollout.

An already-running compatibility-local aggregation can overlap the first Jobs
execution during cutover. Atomic upserts make this safe, but operators should
prefer a controlled restart after the local callback completes.

### Rollback

1. Stop or disable metrics Job producers.
2. Use existing Jobs administration to drain or cancel the review-metrics job
   type.
3. Wait until no matching Jobs remain queued or processing. A running cancelled
   query may still finish its Claims write.
4. Disable the metrics Jobs flag and restart the scheduler in compatibility-local
   mode.

Changing the queue during rollout requires the same barrier across both old and
new queues.

Backlog follow-up `TASK-9935.2` owns Stage 3 removal of the compatibility-local
path after one release of parity evidence. Stage 2B does not leave removal as an
untracked implementation marker.

## Startup And Shutdown

The service remains registered through the existing auxiliary worker lifecycle
and preserves the `claims_review_metrics_task` identity expected by startup
diagnostics and tests.

Startup must not enumerate owners, open Media DBs, or enqueue synchronously before
application readiness. Shutdown stops new APScheduler callbacks, prevents new
fan-out, and awaits or cancels the scheduler wrapper through the existing
lifecycle manager. Jobs already accepted remain owned by Jobs and are not
cancelled by scheduler shutdown.
Stop fan-out cooperatively between pages, owners, and retry delays. An admission
already running in a worker thread may commit after cancellation; retain its
identity and never substitute local execution.

## Observability

Jobs remains the source of truth for queued, processing, retrying, failed,
cancelled, quarantined, and completed work. Existing Jobs APIs and metrics provide
lifecycle visibility.

Claims remains the source of truth for daily domain output. The existing review
metrics endpoint is unchanged.

The scheduler emits one structured summary per callback with:

- normalized slot and date bounds
- owners discovered
- Jobs accepted
- Jobs deduplicated
- owner admission failures
- active producer mode

Per-owner diagnostics may identify the canonical owner ID but do not include DB
paths or raw error text. Worker logs bind operation, safe failure code, Job ID,
and exception type without serializing payload data or stack text into Jobs.

No additional Claims status endpoint, retry endpoint, pause endpoint, queue
dashboard, or administration surface is added.

## Verification Strategy

### Contract Tests

- Accept the exact version 1 payload.
- Reject unknown keys, unsupported versions, booleans and noncanonical owners,
  malformed or non-UTC slots, malformed dates, reversed dates, and windows over
  366 days.
- Reject row/payload owner mismatch.
- Prove payloads and results contain no sensitive or variable-width domain data.

### Scheduling Tests

- Cover the complete flag matrix and immutable startup-mode resolution.
- Verify default, invalid, minimum-interval, and maximum-lookback normalization.
- Verify out-of-range numeric intervals disable only this scheduler safely.
- Verify immediate catch-up does not block startup.
- Verify UTC slot normalization and date-window derivation from one captured
  timestamp.
- Add property-based coverage for slot and date-window invariants, including UTC
  midnight boundaries and leap years.
- Verify `max_instances=1`, coalescing, misfire behavior, and clean shutdown.
- Verify a stall longer than 300 seconds still performs one coalesced repair.
- Verify bounded window-filtered owner pages and nonblocking database/admission
  calls, plus shutdown during a page, owner boundary, and retry delay.
- Verify deterministic owner sorting/deduplication and per-owner failure isolation.
- Verify stable idempotency across duplicate callbacks and process instances.
- Verify transient admission retries reuse one idempotency key and never trigger
  local aggregation.

### Handler And Persistence Tests

- Dispatch the new type through the existing Claims worker.
- Verify SQLite owner-path routing and PostgreSQL shared-backend owner scoping.
- Verify missing SQLite paths are not recreated, including deletion between the
  precheck and actual open; permission/open errors are not successful skips.
- Use the official PostgreSQL fixture with a non-BYPASSRLS role to prove
  multi-owner discovery, team/org/deleted-media aggregation, explicit owner
  filtering, scope restoration, and safe pooled connection reuse.
- Verify exact payload dates reach the aggregation function after delayed
  execution.
- Verify `ok`, `no_activity`, and `owner_database_missing` outcomes.
- Verify sanitized retryable and non-retryable failures.
- Verify atomic insert/update behavior and concurrent duplicate attempts on
  SQLite and the official PostgreSQL fixture.
- Verify overlapping jobs for one owner cannot let an older source snapshot
  overwrite a newer result, while separate PostgreSQL owners remain independent.
- Verify a partial multi-group failure rolls back the complete owner window.
- Verify totals and reason counts use one snapshot under concurrent review writes.
- Verify explicit UTC PostgreSQL grouping and SQLite/PostgreSQL result parity.
- Preserve current reason-code and extractor aggregate behavior.

### Integration And Regression Tests

- Run API-to-Job-to-WorkerSDK-to-Media-DB coverage for one owner.
- Verify retries converge without duplicate daily rows.
- Verify queued cancellation prevents execution and processing cancellation
  rejects stale completion without introducing Claims cancellation controls.
- Preserve compatibility-local scheduler tests.
- Preserve `/api/v1/claims/review/metrics` response and pagination behavior.
- Verify auxiliary startup registration and shutdown.
- Run existing Claims Jobs, analytics export, Jobs manager, lifecycle, and OpenAPI
  regressions affected by shared handler or startup changes.
- Run the complete storage-classifier compatibility matrix for analytics exports.

### Quality Gates

- Focused unit, integration, property, and official PostgreSQL fixture tests.
- Ruff or repository-configured formatting/lint checks on touched files.
- Python compilation checks.
- `git diff --check`.
- Bandit on touched production paths with zero new findings.
- Independent specification-compliance and code-quality review before merge.

## Security And Privacy

- Owner identity is validated at enqueue and execution boundaries.
- Jobs row owner and payload owner must match before DB resolution.
- SQLite paths are derived from canonical owner IDs and never accepted in payloads.
- PostgreSQL maintenance scope is local to trusted background database work;
  aggregation always includes the target owner predicate and resets scope after
  closing its session.
- Payload and result data is bounded, versioned, non-sensitive, and strict.
- Claims producer/handler diagnostics and Jobs errors never expose SQL, paths, connection
  strings, review text, or raw exceptions.
- Queue controls continue to use existing Jobs RBAC.

## Alternatives Considered

### Event-Driven Aggregation

Enqueueing after every review would improve freshness but substantially increase
Jobs volume, change the nightly-delta contract, and still require periodic repair.
It is rejected for Stage 2B.

### Global Aggregation Job

A single Job that processes every owner would reduce Job count but couple owner
failures, weaken cancellation and retry isolation, and make one lease cover an
unbounded fan-out. It is rejected.

### Coordinator And Child Jobs

A durable coordinator could fan out owner Jobs, but Stage 2B has one fixed global
schedule and no dependency graph. Adding coordinator state would duplicate
orchestration concerns without a demonstrated requirement. It is rejected.

### Fixed Nightly UTC Cron

A fixed cron would simplify slot identity but would change the existing interval
configuration and could miss late events without additional backfill behavior. It
is rejected in favor of interval compatibility plus explicit UTC windows.

## Risks And Mitigations

- **Duplicate Jobs:** deterministic identity deduplicates normal replicas and
  restarts; owner-window serialization plus atomic upserts make queue-transition
  or lease-overlap writes safe.
- **Date drift:** payloads carry explicit UTC bounds derived once at enqueue.
- **Jobs outage:** bounded admission retries use the same key; later rolling
  windows repair recent gaps without local fallback ambiguity.
- **High fan-out:** one bounded owner Job isolates work and failures; callbacks
  continue after individual admission failures.
- **Configuration flood:** intervals below 60 seconds are raised and lookback is
  capped at 366 days with warnings.
- **SQLite writer contention:** owner-window transactions are normally two days
  and use the indexed review timestamp; operators are warned before selecting a
  large lookback.
- **Cancellation after dispatch:** documented as best effort; idempotent writes
  may finish even when Jobs rejects late completion.
- **Migration overlap:** explicit producer barriers and rollback instructions
  avoid indefinite dual execution.

## Non-Goals

- Manual historical backfill APIs.
- User-configurable cron schedules.
- Event-driven per-review aggregation.
- Coordinator or child Jobs.
- Claims-owned queue, lease, retry, cancellation, quarantine, or admin controls.
- Review-metrics schema redesign or exact per-day replacement semantics.
- Cluster rebuild migration.
- Removal of the compatibility-local path in Stage 2B.
- Public Claims API response changes.

## Expected Implementation Areas

Implementation planning should expect focused changes in:

- Claims review-metrics domain aggregation module.
- Claims Job contracts, producer helpers, handler dispatch, and worker tests.
- Review-metrics scheduler and auxiliary lifecycle tests.
- Media DB review-metrics persistence helper.
- Environment examples and Claims operations documentation.
- Claims, Jobs, DB, service, property, integration, and PostgreSQL tests.

The implementation must not modify Jobs lifecycle semantics or add Claims queue
administration surfaces merely to support this job type.

## Spec Review

- Placeholder scan: no unresolved decision or implementation markers remain.
- Consistency check: APScheduler owns recurring decisions, Jobs owns lifecycle,
  and Claims owns domain aggregation throughout.
- Compatibility check: interval, lookback, public API, and one-release local mode
  remain available with explicit safety normalization.
- Failure-boundary check: producer admission uncertainty never invokes local
  execution; domain retries remain in WorkerSDK.
- Data check: UTC windows and atomic upserts address the validated correctness
  gaps without changing metric schema semantics.
- Scope check: Stage 2B excludes backfill APIs, cluster work, queue controls,
  schedule customization, and compatibility-path removal.
