# Claims Jobs Stage 2A Analytics Exports Design

Status: Approved for implementation planning

Backlog task: TASK-12989

Related work:

- `Docs/superpowers/specs/2026-06-24-claims-jobs-operational-control-plane-design.md`
- TASK-9935: Claims Jobs operational control plane design
- TASK-9937: Claims Jobs Stage 1 implementation

## Summary

Move Claims analytics export generation onto the shared Jobs control plane when
an export-specific rollout flag is enabled. Claims continues to own export
artifacts, request validation, rendering, and domain outcomes. Jobs continues to
own durable execution, retries, leases, cancellation, quarantine, status, and
administrative controls.

The existing synchronous export behavior remains available while the new flag
is disabled. Stage 2A changes no dashboard queries, review-metrics aggregation,
cluster rebuild behavior, or Jobs administration APIs.

## Context

`POST /api/v1/claims/analytics/export` currently reads monitoring events,
renders JSON or CSV, and stores a ready export during the request. That path can
hold an API worker while querying, filtering, serializing, and persisting a
large result. It also gives operators no durable execution state, retries, or
standard Jobs controls.

Claims Jobs Stage 1 established versioned Claims payload contracts, enqueue
helpers, handler dispatch, and a `WorkerSDK` worker for rebuild and notification
work. Stage 2A extends those patterns for analytics exports without adding a
second queue implementation to Claims.

The broader Stage 2 work is intentionally decomposed:

1. Stage 2A: analytics export Jobs.
2. Stage 2B: review metrics aggregation Jobs.
3. Stage 2C: cluster rebuild Jobs after cluster identity and watchlist-reference
   behavior is designed explicitly.

The three workloads have different triggers, artifacts, retry boundaries, and
idempotency requirements. They should not share one implementation plan.

## Goals

- Return durable, observable export work immediately when Jobs mode is enabled.
- Preserve the current synchronous behavior as a rollout fallback.
- Keep Jobs payloads small, owner-scoped, versioned, and free of filters or
  exported content.
- Make retries deterministic and safe for export artifacts.
- Support shared PostgreSQL and per-user SQLite Media DB deployments.
- Keep Claims export listing and download APIs useful without duplicating Jobs
  lifecycle controls.
- Bound memory and stored-output growth and produce spreadsheet-safe CSV.
- Extract export generation from the oversized `claims_service.py` into a
  focused Claims domain module.

## Non-Goals

- Moving review metrics aggregation or cluster rebuilds onto Jobs.
- Adding pause, resume, drain, retry, cancel, quarantine, prune, or other queue
  controls to Claims endpoints.
- Storing export payloads in Jobs results.
- Moving export payloads into file storage or File Artifacts.
- Adding a Claims cleanup daemon, scheduler, or lease loop.
- Changing Claims dashboard analytics.
- Enabling the new producer flag by default.
- Replacing the existing Claims Jobs worker or `WorkerSDK`.
- Adding request-level `Idempotency-Key` support. Repeated HTTP submissions are
  independent export requests; Jobs idempotency deduplicates enqueue attempts
  only for the same server-generated export artifact.

## Ownership Boundary

Jobs owns:

- Queue persistence and acquisition.
- Job status and retry counters.
- Backoff, leases, cancellation, quarantine, and terminal lifecycle.
- Admin controls, lifecycle events, and operational metrics.
- Efficient scoped reads of Jobs rows, including a reusable batch-read method.

Claims owns:

- Export request normalization and authorization.
- Export artifact rows and artifact state.
- JSON and CSV generation.
- Versioned job-type and payload validation.
- Enqueue helpers and idempotency keys.
- Owner validation and handler dispatch.
- Domain error codes and compact domain results.
- Read-only projection of Jobs status beside Claims artifact state.

Claims must not update Jobs rows directly or expose duplicate queue-control
endpoints. Jobs must not render Claims exports or store export bodies.

## Approaches Considered

### Artifact-First Jobs

Create a queued Claims export artifact, enqueue an ID-only Job, then let the
worker populate the artifact. This preserves the existing list and download
model, keeps large content out of Jobs, and gives the request an export identity
before work begins.

This is the selected approach. Its cross-database dual-write window is handled
with deterministic identifiers, Jobs idempotency, `batch_group`, compensation,
and repair behavior described below.

### Job-Result-Only Exports

Store rendered content in the Jobs result. This removes one artifact transition
but breaks existing Claims export listing and download behavior, duplicates
artifact concerns in Jobs, and risks oversized Jobs rows. Rejected.

### Synchronous Generation With a Shadow Job

Continue rendering in the API process while writing a Job for visibility. This
would report lifecycle state for work that Jobs does not execute and would
duplicate completion behavior. Rejected.

## Feature Flags

The asynchronous producer is enabled only when both settings are true:

- `CLAIMS_JOBS_ENABLED`
- `CLAIMS_ANALYTICS_EXPORT_JOBS_ENABLED`

`CLAIMS_ANALYTICS_EXPORT_JOBS_ENABLED` defaults to false.

`CLAIMS_JOBS_WORKER_ENABLED` remains an independent process-startup setting. A
producer must not require the in-process worker flag because deployments may run
dedicated worker processes.

Analytics export retry count is configured with
`CLAIMS_JOBS_MAX_RETRIES_ANALYTICS_EXPORT` and defaults to three. Jobs applies
the retry count, backoff, quarantine, and lease behavior.

## Module Design

### `claims_analytics_exports.py`

A new focused domain module owns:

- Request filter and pagination normalization.
- Snapshot cutoff calculation.
- JSON and CSV rendering.
- Output-size enforcement.
- Spreadsheet-safe CSV cell encoding.
- Queued artifact creation.
- Conditional artifact transitions.
- Safe artifact error recording.
- Artifact-to-Job reconciliation helpers that use read-only Jobs APIs.

The module accepts an explicit Media DB object and owner ID. It does not create
or manage Jobs leases, retries, or queue controls.

The existing synchronous path and the worker call the same rendering function.
Lifecycle metadata is not part of the rendered content, so equivalent requests
produce equivalent normalized JSON or CSV output.

### Existing Claims Jobs Modules

`claims_job_contracts.py` adds:

- `CLAIMS_GENERATE_ANALYTICS_EXPORT_JOB_TYPE = "claims_generate_analytics_export"`
- A validator for the versioned ID-only payload.
- Stable failure codes and standard `ok` or `skipped` result use.

`claims_jobs.py` adds the feature-flag helper and enqueue helper.

`claims_job_handlers.py` dispatches the new job type, validates row-owner and
payload-owner equality, opens the owner Media DB through the existing
backend-aware factory, and calls the export domain module.

`claims_service.py` retains API authorization and response orchestration but
delegates artifact creation, rendering, status hydration, and download readiness
to the focused export module.

The existing `claims_jobs_worker.py` continues to use `WorkerSDK`. No new worker
loop is introduced.

### Jobs Batch Read

The Jobs module adds an internal, reusable batch-read operation for job IDs with
optional domain and owner filters. Claims always supplies both
`domain="claims"` and the canonical owner ID.

The operation:

- Deduplicates and validates positive integer IDs.
- Uses parameterized queries for SQLite and PostgreSQL.
- Chunks ID lists below backend parameter limits.
- Reads retained archived rows when requested so terminal lifecycle state remains
  available after active-table archival.
- Returns only matching rows.
- Preserves Jobs payload/result decryption behavior.
- Does not add a public API endpoint or administrative behavior.

This avoids one Jobs connection/query per export while keeping lifecycle reads
inside Jobs.

## Persisted Export Model

The existing `claims_analytics_exports` table gains nullable fields:

- `job_id`: SQLite integer or PostgreSQL bigint linking the active Jobs row.
- `error_code`: stable, non-sensitive Claims domain error code.
- `snapshot_at`: UTC request snapshot cutoff.

`job_status` is not a table field. It is projected at read time from Jobs so
Claims does not maintain a second lifecycle state machine.

`error_message` remains for backward compatibility but may contain only a short,
sanitized public message. Raw exceptions are never persisted.

Fresh schemas and upgrade migrations must add the same fields for SQLite and
PostgreSQL. Add an index on `job_id` and retain owner-based indexes. Migration
tests cover both a fresh database and upgrade from the current schema.

### Artifact Status

Claims artifact `status` has these values:

- `queued`: accepted but not yet rendering.
- `processing`: a worker attempt is rendering or persisting output.
- `ready`: a valid export payload is stored.
- `failed`: the latest artifact attempt failed or reconciliation proved that no
  accepted Job exists.

Artifact status is not Jobs lifecycle status. API responses expose a separate
nullable `job_status` projected from Jobs, such as `queued`, `processing`,
`completed`, `failed`, `cancelled`, or `quarantined`.

`ready` is monotonic and terminal for the artifact. Conditional updates prevent
late attempts from changing `ready` back to `processing` or `failed`. An admin
retry of an already-ready Job receives a skipped domain result.

Allowed transitions are:

| From | To | Cause |
| --- | --- | --- |
| `queued` | `processing` | Worker starts or retries generation. |
| `queued` | `failed` | Enqueue compensation or proven orphan. |
| `processing` | `ready` | Payload persisted successfully. |
| `processing` | `failed` | Worker attempt fails. |
| `failed` | `processing` | Jobs retries the same artifact. |
| `ready` | `ready` | Idempotent already-ready observation only. |

All other transitions are rejected by conditional database updates.

## Request Normalization And Snapshot Semantics

Before creating an artifact, Claims:

1. Validates the format as `json` or `csv`.
2. Resolves the authorized target owner.
3. Parses filter timestamps as supported ISO-8601 values and normalizes them to
   UTC.
4. Captures `snapshot_at` in UTC.
5. Uses the earlier of the caller's `end_time` and `snapshot_at` as the effective
   upper bound.
6. Rejects a start time later than the effective end time.
7. Clamps pagination to the existing limit of 1 through 10,000 and a
   non-negative offset.
8. Removes `workspace_id` from persisted filters because the export row and Job
   owner are the only ownership sources.

The event query orders by creation timestamp and a stable ID tie-breaker before
pagination. The fixed snapshot cutoff makes retries deterministic when new
events arrive after request acceptance.

The synchronous fallback uses the same normalization and rendering path. Its
HTTP behavior remains synchronous, but validation is now explicit and shared.

## Asynchronous Create Flow

When both producer flags are enabled:

1. The endpoint authorizes Claims administration access.
2. It resolves a canonical positive integer owner ID. A cross-user
   `workspace_id` requires platform-admin Claims permission.
3. It opens the target owner's Media DB. PostgreSQL uses the shared backend;
   SQLite uses that owner's database path.
4. It performs bounded, best-effort terminal cleanup and orphan reconciliation.
5. It creates a queued export row with a generated `export_id`, normalized
   filters, normalized pagination, format, and `snapshot_at`.
6. It enqueues one Claims Job.
7. It conditionally attaches the returned `job_id` to the export row.
8. It returns HTTP 202 with the export and Job identifiers.

The endpoint returns 202 after Jobs accepts the work even if attaching `job_id`
fails. Returning 503 at that point would encourage clients to create duplicate
exports. The worker and reconciliation path repair the missing association.

If Job creation fails, Claims best-effort marks the artifact failed with
`claims_export_enqueue_failed` and returns 503. It does not generate the export
inline while asynchronous mode is enabled.

## Synchronous Fallback Flow

When either producer flag is disabled:

1. Claims performs the same authorization, owner resolution, request
   normalization, snapshot calculation, rendering, safety checks, and artifact
   persistence.
2. It stores a ready export row without a Job ID.
3. It returns the existing HTTP 200 response behavior.

The response schema makes `job_id` and `job_status` optional so one additive
schema supports both modes.

## Job Contract

The version 1 job payload is exactly:

```json
{
  "version": 1,
  "owner_user_id": "123",
  "export_id": "<validated export id>"
}
```

Unknown keys are rejected. The payload must not include:

- Filters or pagination.
- Monitoring event content.
- Exported JSON or CSV.
- Database paths.
- Workspace objects.
- Credentials, webhook URLs, or other secrets.

`export_id` is the existing server-generated UUID4 hex representation: exactly
32 lowercase hexadecimal characters. This also constrains batch-group values and
download filenames.

The Jobs row uses:

- Domain: `claims`.
- Configured Claims queue.
- Type: `claims_generate_analytics_export`.
- Owner: the same canonical owner as the payload.
- Priority: 5.
- Max retries: `CLAIMS_JOBS_MAX_RETRIES_ANALYTICS_EXPORT`.
- Batch group: `claims-analytics-export:<export_id>`.
- Idempotency key: `claims:analytics_export:<owner_user_id>:<export_id>`.

The result contains only non-sensitive metadata:

```json
{
  "outcome": "ok",
  "export_id": "<export id>",
  "format": "json",
  "event_count": 42,
  "size_bytes": 8192
}
```

An already-ready retry returns a standard skipped result with reason
`already_ready`.

## Worker Flow

For one acquired export Job, the handler:

1. Validates the payload and canonical owner.
2. Requires the Jobs row owner to equal the payload owner.
3. Opens the owner Media DB through the existing backend-aware Media DB factory.
4. Loads the export by both `export_id` and owner ID.
5. Rejects a missing row or ownership mismatch without retry.
6. Returns `already_ready` when the artifact is ready.
7. Repairs a missing `job_id` from the acquired Job ID.
8. Conditionally transitions a non-ready artifact to processing.
9. Reloads and validates the persisted normalized request.
10. Queries events using the stored snapshot cutoff and deterministic ordering.
11. Renders and size-checks the requested format.
12. Conditionally stores the payload and marks the artifact ready.
13. Returns compact result metadata to `WorkerSDK`.

If a late attempt observes that another attempt already made the artifact ready,
it returns `already_ready` instead of overwriting the artifact.

## Export Rendering And Resource Bounds

The existing row limit remains capped at 10,000. Stage 2A also adds
`CLAIMS_ANALYTICS_EXPORT_MAX_BYTES`, defaulting to 10 MiB (10,485,760 bytes).
Invalid or non-positive settings fall back to that default. The limit is
measured against UTF-8 serialized bytes before persistence.

Exceeding the byte limit is non-retryable and records
`claims_export_too_large`. An asynchronous request remains accepted and exposes
the failed artifact through its normal status APIs. A synchronous request
returns HTTP 413 with the same stable code. Jobs results never contain the
rendered payload.

JSON preserves the existing export structure:

- `events`
- `filters`
- `pagination`

CSV preserves stable columns:

- `id`
- `event_type`
- `severity`
- `created_at`
- `payload_json`

CSV uses UTF-8, standard CSV quoting, and spreadsheet-formula protection. String
cells beginning with `=`, `+`, `-`, `@`, tab, or carriage return are prefixed
with a single quote before writing. The download response uses
`text/csv; charset=utf-8` and a safe `Content-Disposition` filename derived only
from the validated export ID.

## List And Download APIs

### Create Response

Jobs-enabled requests return HTTP 202:

```json
{
  "export_id": "<export id>",
  "format": "json",
  "status": "queued",
  "job_id": 456,
  "job_status": "queued",
  "snapshot_at": "<timestamp>",
  "download_url": "/api/v1/claims/analytics/export/<export id>",
  "created_at": "<timestamp>"
}
```

The synchronous fallback returns HTTP 200 with `status="ready"` and null Job
fields.

OpenAPI documents both 200 and 202 responses with the same additive response
model.

### List

The list endpoint reads owner-scoped export rows and hydrates Job statuses with
the scoped Jobs batch-read operation. If Jobs is unavailable, the list still
returns artifact rows with `job_status=null`.

The existing `status` filter continues to mean artifact status. A Job-status
filter is out of scope because it would change pagination semantics across two
stores.

Create and list responses add nullable `job_id`, `job_status`, `error_code`, and
`snapshot_at`. Raw Jobs errors are not exposed.

### Download

Download looks up the export by both owner and export ID.

- Ready JSON returns `application/json`.
- Ready CSV returns `text/csv; charset=utf-8` with a safe attachment filename.
- Non-ready or failed artifacts return HTTP 409 with artifact status, nullable
  Job status, and a stable public code.
- A missing or unauthorized export returns 404 without revealing whether
  another owner has that ID.

A ready artifact is downloadable even if Jobs is unavailable or the Jobs row is
still completing. Valid domain output takes precedence over transient lifecycle
read state.

Pending work uses `claims_export_not_ready`. Terminal Jobs projections use
`claims_export_job_cancelled` or `claims_export_job_quarantined` where
applicable. A failed artifact uses its stored safe code or the generic
`claims_export_failed` fallback.

For platform-admin cross-user exports, generated list and download URLs include
the authorized target `workspace_id` query parameter so per-user SQLite resolves
the correct database. Non-admin callers cannot set another owner.

## Dual-Write Repair And Reconciliation

The Media DB artifact and Jobs row cannot be committed atomically across both
stores. Stage 2A uses these safeguards:

- Server-generated `export_id` is created before either write.
- Jobs idempotency is scoped to owner and export ID.
- Jobs `batch_group` stores the export identity for repair lookup.
- A caught enqueue failure marks the artifact failed.
- A caught attach failure still returns the accepted Job ID.
- The worker repairs `job_id` before rendering.
- Bounded reconciliation repairs queued artifacts by matching domain, owner,
  type, and exact batch group.

`CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC` defines a conservative grace period
and defaults to 300 seconds. Invalid or negative settings fall back to the
default.
An export missing `job_id` becomes failed only when all conditions hold:

1. The grace period has elapsed.
2. A scoped Jobs lookup, including retained archived rows, completed
   successfully.
3. No matching active or retained Job exists.

If Jobs is unavailable, reconciliation leaves the artifact unchanged. It never
converts uncertainty into a failed result.

Reconciliation runs only as bounded best-effort maintenance during existing
create/list activity. It is not a daemon, scheduler, queue, or retry engine.

## Failure Model

### Non-Retryable Domain Failures

- Invalid or unsupported payload version.
- Owner mismatch or non-canonical owner.
- Missing or wrong-owner export artifact.
- Invalid persisted format, filters, pagination, or snapshot.
- Unsupported export format.
- Deterministic JSON or CSV serialization failure.
- Export output above the configured byte limit.

Representative codes include:

- `claims_export_invalid_payload`
- `claims_owner_scope_violation`
- `claims_export_missing`
- `claims_export_invalid_artifact`
- `claims_export_unsupported_format`
- `claims_export_serialization_failed`
- `claims_export_too_large`

### Retryable Failures

- Temporary Media DB connection or transaction failure.
- Temporary storage unavailability.
- Other explicitly classified transient infrastructure failures.

Retryable errors are raised as `ClaimsJobError` with a stable failure code.
Jobs decides whether the Job is queued for retry, failed, or quarantined. Claims
does not inspect retry counters or implement backoff.

On any failed attempt, Claims may record `status="failed"` and a safe code. A
later Jobs retry may transition that artifact back to processing. Cleanup must
therefore consult terminal Jobs state before removing a failed artifact.

Raw exception strings, filters, and export content are not persisted in error
fields or returned to API callers. Logs contain identifiers, operation names,
exception types, and stable codes only.

## Retention And Cleanup

Existing request-time retention cleanup remains bounded and becomes
lifecycle-aware:

- Ready artifacts older than retention may be deleted.
- Failed artifacts with a terminal Jobs status may be deleted after retention.
- Reconciled failed artifacts with no Job may be deleted after retention.
- Failed artifacts whose Jobs row has been pruned may be deleted only after a
  successful Jobs lookup confirms that no active or archived row remains and
  the orphan grace period has elapsed.
- Queued, processing, retrying, or unreconciled artifacts are not deleted solely
  because they are old.
- If Jobs status cannot be read, cleanup skips uncertain non-ready artifacts.

Retention is measured from the terminal artifact update time, not merely initial
creation time.

## Security And Privacy

- Payload validation rejects unknown and sensitive keys.
- Job owner, payload owner, export owner, and requested owner must agree.
- Owner IDs are canonical positive integers before database routing.
- Cross-user access requires platform-admin Claims permission.
- Export lookup includes owner scope at the database query, avoiding existence
  disclosure followed by an authorization check.
- Jobs status hydration includes both Claims domain and owner filters.
- SQL remains parameterized; the Jobs batch read chunks parameter lists safely.
- Jobs results contain no export body or monitoring-event payload.
- CSV protects spreadsheet consumers from formula injection.
- Download filenames derive only from validated server-generated IDs.
- No raw database path is accepted or persisted in a Job.

## PostgreSQL And SQLite Behavior

The existing Media DB runtime factory already selects the configured backend.
Stage 2A does not add another Claims backend selector.

For SQLite, API and worker paths resolve the target owner's Media DB file. For
PostgreSQL, the shared backend is used and every export/event query carries the
explicit owner scope. Migration, owner-scope, and cross-owner denial tests cover
both paths where the repository fixture is available.

## Observability And Administration

Jobs remains the source of truth for queued, processing, retrying, failed,
cancelled, and quarantined execution. Operators use existing Jobs endpoints and
RBAC for controls and detailed lifecycle inspection.

Claims export APIs expose only:

- Artifact identity and artifact status.
- Linked Job ID.
- Read-only Job status.
- Safe domain error code.
- Download readiness and artifact metadata.

The existing Claims dashboard Jobs summary continues to aggregate the Claims
domain. Stage 2A does not add duplicate queue metrics or controls.

## Rollout And Rollback

Rollout order:

1. Deploy schema changes, payload contract, handler, and worker support with the
   producer flag off.
2. Enable Claims Jobs workers and verify they advertise the configured Claims
   queue.
3. Enable `CLAIMS_ANALYTICS_EXPORT_JOBS_ENABLED` for a canary deployment.
4. Verify accepted exports, Jobs status, retries, artifact readiness, downloads,
   and retention behavior.
5. Expand enablement while retaining the synchronous fallback.

Rollback order:

1. Disable the export producer flag so new requests use the synchronous path.
2. Keep Claims Jobs workers running to drain work already accepted.
3. Use existing Jobs admin controls if operators need to pause, cancel, retry,
   quarantine, or drain accepted work.

Disabling workers before producers can strand accepted artifacts and is not the
supported rollback sequence.

## Testing Strategy

### Contract And Enqueue Tests

- Accept the exact versioned ID-only payload.
- Reject unknown keys, filters, content, paths, secrets, invalid IDs, invalid
  owners, and unsupported versions.
- Verify domain, queue, type, owner, priority, retry count, batch group, and
  idempotency key.
- Verify the global and export-specific flag matrix.

### Domain Rendering And Artifact Tests

- Normalize timestamps, filters, pagination, owner, and snapshot cutoff.
- Strip `workspace_id` from persisted filters.
- Produce equivalent normalized content through synchronous and worker paths.
- Preserve deterministic event ordering and fixed snapshot results across
  retries while newer events are inserted.
- Cover JSON and CSV quoting, Unicode, delimiters, newlines, and spreadsheet
  formula prefixes.
- Enforce row and serialized-byte limits.
- Prove conditional transitions cannot move ready back to processing or failed.
- Use property-based state-transition tests for the ready-terminal invariant.

### API Tests

- Jobs disabled returns synchronous HTTP 200 and ready content.
- Jobs enabled returns HTTP 202 with export and Job IDs.
- Enqueue failure records a safe failure and returns 503 without inline work.
- Job attachment failure still returns 202 and is repairable.
- Ready JSON and CSV downloads return correct content types and headers.
- Non-ready and failed downloads return 409 rather than empty success.
- Missing and wrong-owner exports return indistinguishable 404 responses.
- Cross-user platform-admin create, list, and download resolve the target SQLite
  database; non-admin access is denied.
- Jobs lookup failure leaves ready downloads available and returns null Job
  status elsewhere.
- OpenAPI describes both create response statuses and additive fields.

### Worker And Recovery Tests

- Owner mismatch, missing artifact, malformed stored request, unsupported format,
  and oversized output are non-retryable.
- Transient database/storage errors are retryable with stable failure codes.
- Already-ready work returns a skipped result.
- A retry can move failed to processing but cannot overwrite ready.
- Worker startup repair attaches a missing Job ID.
- Reconciliation finds exact batch-group matches, respects the grace period,
  leaves rows unchanged while Jobs is unavailable, and fails only proven
  orphans.
- Cleanup preserves queued, processing, and retrying work and removes only
  eligible terminal artifacts.
- Cancelled, quarantined, missing, and pruned Job projections are represented
  safely.

### Jobs And Database Tests

- Scoped Jobs batch reads work for SQLite and PostgreSQL, chunk large ID lists,
  and never return another owner or domain.
- Export schema creation and migration work from fresh and previous schemas for
  SQLite and PostgreSQL.
- Export create, owner-scoped get, conditional update, list, count, reconcile,
  and cleanup operations remain parameterized and backend-compatible.

### Integration And Regression Tests

- Run API to Jobs to bounded `WorkerSDK` to download end to end with local
  databases and no timing-dependent daemon.
- Run PostgreSQL coverage through the repository isolated fixture; record only
  fixture-reported environment skips.
- Verify Jobs status hydration is bounded and batched without asserting an exact
  internal query count.
- Re-run existing Claims analytics, export cleanup, Stage 1 Claims Jobs, worker
  lifecycle, owner/idempotency, and Jobs lifecycle tests.
- Run Ruff on touched Python files, compile checks, Bandit on touched code,
  `git diff --check`, and the focused pytest suite.

## Risks And Mitigations

Dual-write inconsistency is mitigated by deterministic export IDs, Jobs
idempotency, exact batch groups, compensation, worker repair, and conservative
reconciliation.

Artifact and Job status drift is mitigated by exposing separate states and
keeping Jobs authoritative for lifecycle.

Retry races are mitigated by owner-scoped reads, conditional writes, and a
terminal ready state.

Snapshot drift is mitigated by a persisted UTC cutoff and deterministic ordering.

Memory and database growth are mitigated by row limits, byte limits, compact
Jobs results, and lifecycle-aware retention.

Cross-owner access is mitigated by canonical owner validation and owner scope in
API, artifact, Jobs, and Media DB operations.

Spreadsheet formula execution is mitigated by CSV cell neutralization and safe
download headers.

Rollout incompatibility is mitigated by an export-specific opt-in flag, additive
response fields, a synchronous fallback, and producer-first rollback.

Client retries without a request-level idempotency key may create more than one
valid export artifact if the original HTTP response is lost. This is existing
create-request behavior, is non-destructive, and is kept explicit rather than
conflated with Jobs idempotency for one artifact.

## Acceptance Criteria

- Jobs-enabled export requests return HTTP 202 with a durable Job and queued
  artifact.
- Jobs-disabled export requests retain synchronous HTTP 200 behavior.
- Worker retries are deterministic and cannot overwrite a ready artifact.
- Jobs payloads and results contain no filters or export content.
- Claims owns artifact behavior only; Jobs owns lifecycle and admin controls.
- SQLite and PostgreSQL owner scope is explicit and tested.
- Pending and failed downloads never return empty HTTP 200 responses.
- Output row count, byte size, CSV safety, and retention are bounded.
- Dual-write interruptions are repairable without a Claims queue implementation.
- No review-metrics, cluster-rebuild, scheduler, or queue-control work enters
  Stage 2A.

## Spec Review

- Placeholder scan: no placeholders remain.
- Internal consistency: Claims owns artifacts and rendering throughout; Jobs
  owns execution lifecycle and controls throughout.
- Scope check: Stage 2A covers one user-visible workflow and one supporting Jobs
  batch-read primitive. Review metrics and cluster rebuilds remain separate.
- Ambiguity check: flags, HTTP statuses, ownership, payload fields, artifact and
  Job states, snapshot cutoff, idempotency, dual-write repair, retry classes,
  cleanup, resource limits, CSV safety, rollout, and rollback are explicit.
- Testing coverage: contracts, APIs, domain rendering, recovery, migrations,
  both database backends, integration, regressions, lint, compile, and security
  checks are specified.
