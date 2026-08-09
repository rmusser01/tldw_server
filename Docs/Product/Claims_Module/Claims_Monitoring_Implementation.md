# Claims Monitoring Implementation Plan

## Goals
- Expose claims monitoring configuration via API for UI consumption.
- Emit claims provider, rebuild, and review metrics through the existing metrics registry.
- Provide lightweight health and alerting endpoints that can be wired to dashboards.

## Data Model
- `ClaimsMonitoringConfig`
  - `id` (PK), `workspace_id` (TEXT), `threshold_ratio` (REAL), `baseline_ratio` (REAL),
    `slack_webhook_url` (TEXT), `webhook_url` (TEXT), `email_recipients` (TEXT JSON),
    `enabled` (BOOLEAN), `created_at`, `updated_at`.
- `ClaimsMonitoringAlerts` (alert rules)
  - `id` (PK), `workspace_id` (TEXT), `name` (TEXT), `alert_type` (TEXT),
    `threshold_ratio` (REAL), `baseline_ratio` (REAL), `channels` (TEXT JSON),
    `enabled` (BOOLEAN), `created_at`, `updated_at`.
- `ClaimsMonitoringEvents`
  - `id` (PK), `workspace_id` (TEXT), `event_type` (TEXT), `severity` (TEXT),
    `payload_json` (TEXT), `created_at`.
- `ClaimsMonitoringHealth`
  - `id` (PK), `workspace_id` (TEXT), `queue_size` (INTEGER),
    `last_worker_heartbeat` (TIMESTAMP), `last_failure_at` (TIMESTAMP),
    `last_failure_reason` (TEXT), `updated_at`.

For v1, `workspace_id` maps to the current user id (string). Multi-tenant org/team
extensions should add org/team identifiers in later iterations.

## Alert Evaluation Semantics
The monitoring pipeline computes two ratios for alert evaluation:
- `window_ratio`: unsupported claims ratio over the evaluation window.
- `baseline_window_ratio`: unsupported claims ratio over the baseline window.

Alert configuration fields map to those ratios as follows:
- `threshold_ratio` is an absolute ceiling for `window_ratio`. Alerts fire when
  `window_ratio > threshold_ratio`.
- `baseline_ratio` (config) is a drift threshold (delta). If set (non-null or
  > 0.0), alerts also fire when
  `window_ratio - baseline_window_ratio > baseline_ratio`.

`threshold_ratio` is required for threshold-based alert types; `baseline_ratio`
is optional and only enables drift checks. Alert types may require one or both
ratios; if a ratio is omitted (null), its check is skipped. Example: if
`baseline_window_ratio = 0.08`, `threshold_ratio = 0.20`, and
`baseline_ratio = 0.05`, the alert triggers when `window_ratio > 0.20` or
`window_ratio - 0.08 > 0.05`.

## Metrics
Register in a dedicated claims monitoring module:
- `claims_provider_requests_total` (counter) labels: provider, model, mode.
- `claims_provider_latency_seconds` (histogram) labels: provider, model.
- `claims_provider_errors_total` (counter) labels: provider, model, reason.
- `claims_provider_estimated_cost_usd_total` (counter) labels: provider, model.
- `claims_rebuild_queue_size` (gauge).
- `claims_rebuild_processed_total` (counter).
- `claims_rebuild_failed_total` (counter).
- `claims_rebuild_job_duration_seconds` (histogram).
- `claims_rebuild_worker_heartbeat_timestamp` (gauge).
- `claims_review_queue_size` (gauge).
- `claims_review_processed_total` (counter).
- `claims_review_latency_seconds` (histogram).
- `claims_alert_webhook_delivered_total` (counter) labels: status (`success`, `failure`).
- `claims_alert_webhook_failed_total` (counter) labels: reason (`timeout`, `dns`, `tls`, `http_4xx`, `http_5xx`, `invalid_url`, `other`).
- `claims_alert_webhook_latency_seconds` (histogram) labels: status (`success`, `failure`).

## Review Metrics Aggregation
- Persist nightly review deltas to `ClaimsReviewExtractorMetricsDaily` (workspace_id maps to user id).
- Aggregates `claims_review_log` into per-extractor daily metrics with correction motifs.
- Scheduler controls:
  - `CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED`
  - `CLAIMS_REVIEW_METRICS_INTERVAL_SEC`
  - `CLAIMS_REVIEW_METRICS_LOOKBACK_DAYS`

## API Surface
- `GET /api/v1/claims/monitoring/config`
- `PATCH /api/v1/claims/monitoring/config`
- `GET /api/v1/claims/alerts`
- `POST /api/v1/claims/alerts`
- `PATCH /api/v1/claims/alerts/{alert_id}`
- `DELETE /api/v1/claims/alerts/{alert_id}`
- `GET /api/v1/claims/rebuild/health`
- `GET /api/v1/claims/review/metrics`
- `POST /api/v1/claims/analytics/export`
- `GET /api/v1/claims/analytics/export/{export_id}`
- `GET /api/v1/claims/analytics/exports`

The monitoring config and alerts are stored in Media DB for now.

### Endpoint Semantics + Schemas
All endpoints are scoped to the current workspace (v1: user id). Non-admin users
are restricted to their own `workspace_id` and receive a 403 if they attempt to
access another workspace's data. Admin users may query any `workspace_id` for
multi-tenant access; an admin's own `workspace_id` still maps to their user id
unless they explicitly request another workspace. Unless noted, responses
include `created_at`/`updated_at` timestamps (ISO 8601).

#### GET /api/v1/claims/monitoring/config
Returns the single config row for the current workspace.

Response schema:
```json
{
  "id": "string",
  "workspace_id": "string",
  "threshold_ratio": 0.0,     // float, nullable: false, min: 0.0; alert when window_ratio > threshold_ratio
  "baseline_ratio": 0.0,      // float, nullable: false, min: 0.0; drift delta, alert when window_ratio - baseline_window_ratio > baseline_ratio (0 disables drift)
  "slack_webhook_url": "string|null", // nullable: true, https URL
  "webhook_url": "string|null",       // nullable: true, https URL
  "email_recipients": ["string"],     // array of email strings, nullable: true
  "enabled": true,
  "created_at": "string",
  "updated_at": "string"
}
```

#### PATCH /api/v1/claims/monitoring/config
Creates or updates the single config row for the current workspace.

Request schema (all optional; patchable fields):
```json
{
  "threshold_ratio": 0.0,
  "baseline_ratio": 0.0,
  "slack_webhook_url": "string|null",
  "webhook_url": "string|null",
  "email_recipients": ["string"],
  "enabled": true
}
```
Constraints: `baseline_ratio <= threshold_ratio`, ratios >= 0.0; webhook URLs must be https.
`threshold_ratio` is the absolute `window_ratio` ceiling; `baseline_ratio` is an
optional drift threshold (delta above `baseline_window_ratio`). Use
`baseline_ratio = 0.0` to disable drift checks for the workspace config.

Response schema: same as GET `/claims/monitoring/config`.

Example: configure email recipients + enable digests (no SMTP required):
```json
{
  "email_recipients": ["alerts@example.com"],
  "enabled": true
}
```
Digest delivery is controlled via environment variables:
- `CLAIMS_ALERT_EMAIL_DIGEST_ENABLED=true`
- `CLAIMS_ALERT_EMAIL_DIGEST_INTERVAL_SEC=86400`
- `CLAIMS_ALERT_EMAIL_DIGEST_MAX_EVENTS=500`

If you do not have SMTP configured, keep `EMAIL_PROVIDER=mock` to log or write
emails locally (`EMAIL_MOCK_OUTPUT=console|file|both`).

#### GET /api/v1/claims/alerts
Lists alert rules for the current workspace.

Query params:
```json
{
  "limit": 100,                     // optional, max 1000
  "offset": 0,                      // optional
  "sort_by": "created_at|name|alert_type", // optional
  "sort_order": "asc|desc"          // optional
}
```

Response schema:
```json
{
  "items": [
    {
      "id": "string",
      "workspace_id": "string",
      "name": "string",
      "alert_type": "string",          // e.g., "threshold_breach", "provider_error_rate"
      "threshold_ratio": 0.0,          // nullable: true, min: 0.0; alert when window_ratio > threshold_ratio
      "baseline_ratio": 0.0,           // nullable: true, min: 0.0; drift delta, alert when window_ratio - baseline_window_ratio > baseline_ratio
      "channels": {
        "slack": true,
        "webhook": true,
        "email": true
      },                               // nullable: false, at least one channel true
      "enabled": true,
      "created_at": "string",
      "updated_at": "string"
    }
  ],
  "total": 0,
  "limit": 100,
  "offset": 0
}
```

#### POST /api/v1/claims/alerts
Creates an alert rule (not an event record). Events are written by the monitoring
pipeline into `ClaimsMonitoringEvents`.

Request schema (required fields: `name`, `alert_type`, `channels`):
```json
{
  "name": "string",                   // non-empty
  "alert_type": "string",
  "threshold_ratio": 0.0,             // optional; alert when window_ratio > threshold_ratio
  "baseline_ratio": 0.0,              // optional; drift delta, alert when window_ratio - baseline_window_ratio > baseline_ratio
  "channels": {
    "slack": true,
    "webhook": true,
    "email": true
  },
  "enabled": true
}
```
Constraints: `baseline_ratio <= threshold_ratio` when both provided; ratios >= 0.0.
`threshold_ratio` is the absolute `window_ratio` ceiling; `baseline_ratio` is an
optional drift threshold (delta above `baseline_window_ratio`). If a ratio is
null, its corresponding check is skipped.
at least one channel must be true (otherwise 400 `invalid_channels`).

Response schema: single alert rule (same shape as GET list item).

#### PATCH /api/v1/claims/alerts/{alert_id}
Updates an alert rule.

Request schema (patchable fields):
```json
{
  "name": "string",
  "alert_type": "string",
  "threshold_ratio": 0.0,             // optional; alert when window_ratio > threshold_ratio
  "baseline_ratio": 0.0,              // optional; drift delta, alert when window_ratio - baseline_window_ratio > baseline_ratio
  "channels": { "slack": true, "webhook": false, "email": true },
  "enabled": true
}
```
Constraints: `baseline_ratio <= threshold_ratio` when both provided; ratios >= 0.0.
`threshold_ratio` is the absolute `window_ratio` ceiling; `baseline_ratio` is an
optional drift threshold (delta above `baseline_window_ratio`). If a ratio is
null, its corresponding check is skipped.
at least one channel must be true (otherwise 400 `invalid_channels`).
Response schema: single alert rule.

#### DELETE /api/v1/claims/alerts/{alert_id}
Deletes an alert rule.

Response schema:
```json
{ "deleted": true }
```

#### GET /api/v1/claims/rebuild/health
Returns persisted service health for claims rebuild workers.

Response schema:
```json
{
  "workspace_id": "string",
  "queue_size": 0,                    // integer, >= 0
  "last_worker_heartbeat": "string|null",
  "last_failure_at": "string|null",
  "last_failure_reason": "string|null",
  "updated_at": "string"
}
```

#### POST /api/v1/claims/analytics/export
Creates an export for claims monitoring analytics.

Request schema:
```json
{
  "format": "csv|json",               // required
  "filters": {
    "workspace_id": "string|null",    // optional admin cross-owner target; non-admins may omit or provide their own ID; 403 for another workspace
    "event_type": "string|null",
    "severity": "string|null",
    "provider": "string|null",
    "model": "string|null",
    "start_time": "string|null",      // ISO 8601
    "end_time": "string|null"         // ISO 8601
  },
  "pagination": {
    "limit": 1000,                    // optional, max 10000
    "offset": 0                       // optional
  }
}
```

Response schema:
```json
{
  "export_id": "string",
  "format": "csv|json",
  "status": "queued|processing|ready|failed",
  "job_id": "integer|null",
  "job_status": "queued|processing|completed|failed|cancelled|quarantined|null",
  "error_code": "string|null",
  "snapshot_at": "string|null",
  "download_url": "string|null",
  "created_at": "string"
}
```
When either `CLAIMS_JOBS_ENABLED` or `CLAIMS_ANALYTICS_EXPORT_JOBS_ENABLED` is
disabled, the request runs synchronously and returns HTTP 200 with a ready
artifact and null Job fields. When both producer flags are enabled, the request
returns HTTP 202 with a queued artifact, `job_id`, `job_status`, and
`snapshot_at`; these additive fields remain nullable in the response contract.
If the enqueue call returns without a complete Job ID/status projection, Claims
still returns HTTP 202 because the Job may already be accepted. The worker and
bounded reconciliation repair the artifact association. HTTP 503 with
`claims_export_enqueue_failed` applies only when enqueue raises before returning
acceptance, never after accepted Job creation. The producer does not require
`CLAIMS_JOBS_WORKER_ENABLED`, because dedicated worker processes may run the
WorkerSDK service separately.

Download endpoint: `GET /api/v1/claims/analytics/export/{export_id}` returns the
export payload (JSON) or CSV body only when the artifact is ready.

#### GET /api/v1/claims/analytics/exports
Lists stored analytics exports.

Query params:
```json
{
  "limit": 100,                       // optional, max 1000
  "offset": 0,                        // optional
  "status": "queued|processing|ready|failed", // optional
  "format": "csv|json",               // optional
  "workspace_id": "string|null"       // optional, admin-only filter; non-admins are scoped to their workspace_id
}
```
Non-admin users are always scoped to their own workspace and receive a 403 if
they attempt to query another workspace via `workspace_id`.

Response schema:
```json
{
  "exports": [
    {
      "export_id": "string",
      "format": "csv|json",
      "status": "queued|processing|ready|failed",
      "job_id": "integer|null",
      "job_status": "queued|processing|completed|failed|cancelled|quarantined|null",
      "error_code": "string|null",
      "snapshot_at": "string|null",
      "download_url": "string|null",
      "created_at": "string",
      "updated_at": "string",
      "filters": "object|null",
      "pagination": "object|null",
      "error_message": "string|null"
    }
  ],
  "total": 0,
  "limit": 100,
  "offset": 0,
  "pagination": {
    "mode": "offset",
    "limit": 100,
    "offset": 0,
    "total": 0,
    "has_more": false,
    "next_offset": null
  },
  "has_more": false,
  "next_offset": null
}
```

Export requests are bounded to at most 10,000 monitoring-event rows. Rendered
JSON or CSV output is bounded by `CLAIMS_ANALYTICS_EXPORT_MAX_BYTES`; the default
configured limit is 10 MiB (10,485,760 UTF-8 bytes). Any configured positive
integer value overrides that default, while invalid or non-positive values use 10 MiB.
At artifact acceptance, Claims also records an internal owner-scoped monitoring
event-ID high-water. Rendering and retries apply that high-water with the
timestamp cutoff so events added later with equal or backdated timestamps are
excluded. This internal value is not exposed in the API or Jobs contracts; the
matching `(user_id, created_at, id)` index is present on SQLite and PostgreSQL.
Synchronous requests that exceed the configured byte limit return HTTP 413 with
the stable `claims_export_too_large` code; asynchronous requests expose the safe
failed artifact through the normal status APIs.

CSV downloads use UTF-8, standard CSV quoting, and spreadsheet-formula
protection. String cells beginning with `=`, `+`, `-`, `@`, tab, or carriage
return are prefixed with a single quote before writing. Ready CSV responses use
`text/csv; charset=utf-8` and attachment headers with a safe filename derived
only from the validated server-generated `export_id`. JSON responses use
`application/json`.

Export lookup, list, and download are owner-scoped at the database query. A
platform-admin caller authorized to target another user receives generated
cross-user URLs with that target's `workspace_id` query parameter so per-user
SQLite routing resolves the correct database. Non-admin callers cannot set
another owner, and missing and wrong-owner exports both return HTTP 404.

### Export Status Lifecycle
- **queued**: export request accepted and queued for async processing.
- **processing**: a worker attempt is rendering or persisting output.
- **ready**: export payload stored; `download_url` populated.
- **failed**: the latest artifact attempt failed or reconciliation proved that
  no accepted Job exists; `error_message` is a short safe message.

Claims artifact status and Jobs lifecycle status are separate. `status` is the
artifact state owned by Claims. `job_status` is a read-only projection of the
shared Jobs lifecycle, which may include `queued`, `processing`, `completed`,
`failed`, `cancelled`, or `quarantined`. Jobs remains authoritative for retries,
leases, cancellation, quarantine, and terminal execution state; Claims does not
maintain a second Jobs state machine. A ready artifact remains downloadable even
when its Jobs row is unavailable or still completing.

State transitions:
- queued -> processing -> ready: a worker completes payload generation.
- queued -> failed: enqueue compensation or reconciliation proves no accepted
  Job exists.
- processing -> failed: any worker attempt fails, including a retryable failure.
- failed -> processing: Jobs later retries the same artifact.

Client guidance:
- Poll `GET /api/v1/claims/analytics/export/{export_id}` with exponential backoff
  (e.g., 2s, 5s, 10s, 30s) until `status` is `ready` or `failed`.
- No webhook callback is provided; `export_id` is the tracking identifier.

Cleanup:
- Retention cleanup and orphan reconciliation run as bounded, best-effort work
  during existing Claims create/list activity. They use
  `CLAIMS_ANALYTICS_EXPORT_RETENTION_HOURS` (default 24) and
  `CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC` (default 300).
- Once retention eligibility is reached, lifecycle-aware cleanup may delete aged
  ready artifacts, aged failed artifacts whose Jobs state is terminal, and
  reconciled failed artifacts with no Job. A failed artifact that still retains
  `job_id` becomes deletion-eligible only after retention plus orphan grace when
  a successful owner-scoped lookup of active and archived Jobs returns no row,
  proving that the Job was pruned or is missing. Queued, processing, retrying,
  and uncertain non-ready artifacts are preserved. A Jobs lookup failure or
  owner/domain/type mismatch remains uncertainty and preserves the artifact.
- This is request-time maintenance, not a scheduled Claims cleanup job,
  scheduler, daemon, lease loop, or retry engine. Clients should download before
  expiry.

### Error Handling

Claims analytics export endpoints use FastAPI `detail` responses. These shapes
apply only to the analytics export endpoints and do not redefine unrelated
Claims endpoint errors.

A missing or wrong-owner export returns the same generic HTTP 404 response:

```json
{
  "detail": "Export not found"
}
```

Non-ready and failed downloads return HTTP 409:

```json
{
  "detail": {
    "code": "claims_export_not_ready",
    "status": "queued",
    "job_status": null
  }
}
```

When asynchronous mode is enabled, an enqueue exception before acceptance
returns HTTP 503:

```json
{
  "detail": {
    "code": "claims_export_enqueue_failed",
    "message": "Claims analytics export could not be queued."
  }
}
```

An incomplete projection returned by enqueue is not a 503 condition. Claims
returns HTTP 202 because the Job may already be accepted, and the worker or
bounded reconciliation repairs the association. Claims never returns HTTP 503
after accepted Job creation. Unrelated Claims endpoint behavior is unchanged;
for example, alert validation still uses `invalid_channels` when all channels
are false.

Download behavior is explicit: a non-ready or failed artifact returns HTTP 409
with artifact status, nullable `job_status`, and a stable public error code:

- Pending or retrying work uses `claims_export_not_ready`.
- A cancelled Jobs projection uses `claims_export_job_cancelled`.
- A quarantined Jobs projection uses `claims_export_job_quarantined`.
- A failed artifact uses its stored safe `error_code`, or
  `claims_export_failed` when no safe stored code is available.

The generic 404 response does not reveal whether another owner has the requested
export ID.

Webhook delivery:
- Retry strategy: exponential backoff with jitter.
- Max retries: 5 attempts (initial attempt + 4 retries).
- Backoff schedule: 5s, 15s, 45s, 120s, 300s (cap at 5 minutes).
- On non-2xx response or network error, record a failed delivery event with
  reason and attempt count; log at warn and emit `claims_alert_webhook_failed_total`
  and `claims_alert_webhook_delivered_total{status="failure"}`.
- On success (2xx), record delivery success, log at info, emit
  `claims_alert_webhook_delivered_total{status="success"}`, and record
  `claims_alert_webhook_latency_seconds` with the observed duration.
- Reason/status mapping guidance:
  - `status="success"`: any 2xx response.
  - `status="failure"`: any non-2xx response or network/validation error.
  - `reason="http_4xx"`: response status 400-499.
  - `reason="http_5xx"`: response status 500-599.
  - `reason="timeout"`: connect/read timeout exceeded.
  - `reason="dns"`: name resolution failure.
  - `reason="tls"`: TLS handshake/verification error.
  - `reason="invalid_url"`: URL validation fails (scheme/host/SSRF policy).
  - `reason="other"`: fallback for uncategorized errors.

### Health Persistence
Health endpoints must read from persisted state in Media DB so multi-instance
restarts do not reset queue/heartbeat visibility. Workers update
`ClaimsMonitoringHealth` on each heartbeat and queue size change; the API reads
the latest row per workspace.

## Access Control
- Require `admin` role or `claims.admin` permission for config/alerts endpoints.
- Health endpoint should be limited to `admin`/SRE roles.

## Jobs Operator Boundary
Claims owns export authorization, request normalization, artifact status,
rendering, reconciliation, and download readiness. Shared Jobs owns queue
admission, leases, retries, backoff, cancellation, quarantine, lifecycle
status, and administrative controls. Existing Jobs admin endpoints are the only
pause, cancel, retry, quarantine, or drain controls; Claims must not recreate
those controls or add a Claims scheduler.

Producer and worker enablement are independent. The asynchronous producer
requires both `CLAIMS_JOBS_ENABLED` and
`CLAIMS_ANALYTICS_EXPORT_JOBS_ENABLED`, while
`CLAIMS_JOBS_WORKER_ENABLED` only controls whether the WorkerSDK service starts
in that process. Request-time reconciliation and cleanup are bounded maintenance
on existing create/list activity, not a Claims scheduler, daemon, or retry loop.

## Rollout And Rollback
Roll out in this order:

1. Deploy the schema and Job payload/handler support with producers disabled.
2. Start or enable Claims WorkerSDK workers and verify they advertise the
   configured Claims queue.
3. Enable `CLAIMS_ANALYTICS_EXPORT_JOBS_ENABLED` for a producer canary while
   retaining the synchronous fallback elsewhere.
4. Verify accepted exports, Job status, retries, artifact readiness, safe
   downloads, limits, and retention behavior before expanding the canary.

Roll back in this order:

1. Disable `CLAIMS_ANALYTICS_EXPORT_JOBS_ENABLED` first so new analytics export
   requests use the synchronous path.
2. Keep `CLAIMS_JOBS_ENABLED` unchanged unless separately rolling back all Claims
   Jobs workloads, because it also controls Stage 1. Keep workers running to
   drain accepted Jobs and repair accepted artifacts.
3. Disable workers only after the accepted Jobs have drained. Use the existing
   Jobs admin endpoints for any required pause, cancel, retry, quarantine, or
   drain action.

Disabling workers before the analytics export producer can strand accepted
artifacts and is not the supported rollback sequence.

## Testing
- API tests for config CRUD and rebuild health response shape.
- Unit tests for metric registration and alert config serialization.
- Config CRUD cases: reject invalid `webhook_url`/`slack_webhook_url`, enforce
  `workspace_id` constraints on create/update, and return 404 on deleting
  non-existent configs.
- Alert threshold edge cases: validate `baseline_ratio <= threshold_ratio` and
  reject negative ratios.
- Integration coverage: webhook delivery retries and end-to-end alert emission
  from threshold breach to stored `ClaimsMonitoringEvents`.
