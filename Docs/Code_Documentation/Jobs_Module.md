# Jobs Module

The Jobs module provides a reusable, DB-backed job queue with leasing, retries, cancellation, and metrics. It is domain-agnostic and used by Chatbooks and Prompt Studio; other subsystems can adopt it incrementally.

## Standard Queues

- default
- high
- low

Use these names across domains to keep operations consistent.

## Connectors Domain

The external connectors stack now uses the Jobs module for user-visible and recurring file sync work.

- Domain: `connectors`
- Queue: `default`
- Current job types:
  - `import`
  - `incremental_sync`
  - `subscription_renewal`
  - `repair_rescan`
- Trigger paths:
  - connector source creation can enqueue `import`
  - `POST /api/v1/connectors/sources/{source_id}/sync` enqueues `incremental_sync`
  - provider webhooks enqueue `incremental_sync` after validation and dedupe
  - the recurring connectors sync scheduler enqueues `incremental_sync`, `subscription_renewal`, or `repair_rescan`

### Source-Scoped Fencing

Connector sync jobs are not only deduped at Jobs level. They also reserve a source-scoped fence in `external_source_sync_state`.

- `active_job_id` and `active_job_started_at` track the current source owner
- duplicate submissions for the same source return the already-active job when it is still live
- stale fences are cleared before reserving a new job
- workers renew their Jobs lease and update source sync state while running
- file-hosting sync jobs report `processed`, `skipped`, `failed`, and `degraded` counts in the job result

## Shared Workspace Clone Domain

Shared Workspace cloning uses Jobs as the sole operation-status authority while
the recipient-facing API projects a bounded Workspace operation envelope.

- Domain: `sharing`
- Queue: `workspace-clone`
- Job type: `workspace_clone`
- Owner: authenticated recipient user ID
- Automatic retries: disabled (`max_retries=0`)
- Durable replay: an owner-scoped receipt retains the accepted operation
  correlation for 31 days across active and archived Jobs

Recipients use only
`POST /api/v1/sharing/shared-with-me/{share_id}/clone` and the returned
`poll_href`; generic Jobs administration rows are not part of the recipient
contract. The worker keeps staged media and the target Workspace hidden until
fenced Job completion and final publication. A bounded reconciliation loop
finishes publication or cleanup after a process exit. Copy success reports
text, citation, and vector readiness independently; `needs_indexing` is a
successful copy with incomplete vector readiness, not an indexing success.
Each copy supports at most 10,000 unique Media records and rejects a larger
snapshot before Media loading or target reservation.

The in-process worker is enabled by default when the Sharing route is enabled,
unless `SHARED_WORKSPACE_CLONE_JOBS_WORKER_ENABLED=false` or
`TLDW_WORKERS_SIDECAR_MODE=true`. In sidecar mode run exactly one standalone
consumer:

```bash
python -m tldw_Server_API.app.core.Sharing.shared_workspace_clone_jobs_worker
```

Operational tuning uses `SHARED_WORKSPACE_CLONE_JOBS_WORKER_ID`,
`SHARED_WORKSPACE_CLONE_JOBS_LEASE_SECONDS`,
`SHARED_WORKSPACE_CLONE_JOBS_RENEW_JITTER_SECONDS`,
`SHARED_WORKSPACE_CLONE_JOBS_RENEW_THRESHOLD_SECONDS`,
`SHARED_WORKSPACE_CLONE_JOBS_BACKOFF_BASE_SECONDS`,
`SHARED_WORKSPACE_CLONE_JOBS_BACKOFF_MAX_SECONDS`,
`SHARED_WORKSPACE_CLONE_COMPLETION_TIMEOUT_SECONDS`,
`SHARED_WORKSPACE_CLONE_AUTHORIZATION_TIMEOUT_SECONDS`, and
`SHARED_WORKSPACE_CLONE_RECONCILE_SECONDS`.
`SHARED_WORKSPACE_CLONE_VECTOR_RETRIEVAL_CONFIGURED=true` tells the result
projection that vector retrieval is expected, so a copied workspace without
vectors reports `needs_indexing`; it does not enqueue or perform indexing. The
queue name is fixed; do not configure a second queue or run both application
and sidecar consumers.

## Audio Studio Domain

Audio Studio uses Jobs for user-visible audio asset work.

- Domain: `audio_studio`
- Queue: `default`
- Endpoint-enqueued job types:
  - `audio_studio_generate`
  - `audio_studio_render`
  - `audio_studio_export`
- Trigger paths:
  - `POST /api/v1/audio-studio/projects/{project_id}/generations`
  - `POST /api/v1/audio-studio/projects/{project_id}/renders`
  - `POST /api/v1/audio-studio/projects/{project_id}/exports`

Audio Studio job payloads are sanitized before persistence. Client requests must not include provider secrets or external URLs; provider base URLs are resolved from environment configuration and checked against `AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST`. Jobs are idempotent by project, target resource, target revision, job type, and caller idempotency key. Audiobook migration commit currently runs as a server-side transaction through `POST /api/v1/audio-studio/migrations/audiobook/commit`; the worker has a reserved migration handler for a future async migration path, but the public endpoint does not enqueue that job type today.

### Recurring Connector Scheduler

`tldw_Server_API/app/services/connectors_sync_scheduler.py` is an APScheduler bridge that scans sources and enqueues Jobs. It does not perform sync work itself.

- `CONNECTORS_SYNC_SCHEDULER_ENABLED=true` starts the service at app startup
- `CONNECTORS_SYNC_SCHEDULER_SCAN_SEC` controls scan cadence (default `300`)
- `CONNECTORS_SYNC_RENEWAL_LOOKAHEAD_SEC` controls how early expiring webhook subscriptions are renewed (default `3600`)

## Admin Quick Reference

- Safe reads
  - `GET /api/v1/jobs/stats` - aggregated counts by `{domain,queue,job_type}` (queued vs scheduled vs processing vs quarantined)
  - `GET /api/v1/jobs/list` - list jobs with filters (`domain, queue, status, owner_user_id, job_type, limit`, sorting)
  - `GET /api/v1/jobs/events` - outbox polling (`after_id, limit, domain, queue, job_type`)
  - `GET /api/v1/jobs/events/stream` - SSE cursor stream (`after_id`)
  - `GET /api/v1/jobs/queue/status` - `{ paused, drain }` for a queue
  - `GET /api/v1/jobs/{job_id}/attachments` - list attachments/logs (optionally scoped with `domain` for RBAC/RLS)
  - `GET /api/v1/jobs/sla/policies` - SLA policies (optionally filtered; respects domain RBAC when enabled)
  - `GET /api/v1/jobs/archive/meta` - archive compression metadata for a job (if archived; optionally scoped with `domain` for RBAC/RLS)

- Admin writes (require `X-Confirm: true` unless `dry_run: true`)
  - `POST /api/v1/jobs/prune` - delete old terminal jobs (supports `dry_run` and `detail_top_k`)
  - `POST /api/v1/jobs/ttl/sweep` - cancel/fail queued-by-age and processing-by-runtime
    - RBAC special-case: with `JOBS_DOMAIN_SCOPED_RBAC` + `JOBS_RBAC_FORCE` and domain provided, returns `{affected:0}` without `X-Confirm`
  - `POST /api/v1/jobs/batch/cancel` - cancel queued/processing (scoped)
  - `POST /api/v1/jobs/batch/reschedule` - delay or set-now queued jobs (scoped)
  - `POST /api/v1/jobs/batch/requeue_quarantined` - move quarantined back to queued (scoped)
  - `POST /api/v1/jobs/{job_id}/attachments` - add attachment/log (optionally scoped with `domain` for RBAC/RLS)
  - `POST /api/v1/jobs/sla/policy` - upsert per-job_type SLA policy (domain-scoped when domain RBAC is enabled)
  - `POST /api/v1/jobs/queue/control` - `{ action: 'pause'|'resume'|'drain' }`
  - `POST /api/v1/jobs/crypto/rotate` - re-encrypt encrypted fields (supports `dry_run`, requires `X-Confirm` to execute)

> Destructive Ops
> - Always include `X-Confirm: true` for prune, TTL sweep, batch cancel/reschedule/requeue, and crypto rotate unless running `dry_run`.
> - Scope operations with `domain` (and optionally `queue`/`job_type`) to avoid wide-impact actions.
> - With domain-scoped RBAC enabled and forced (`JOBS_DOMAIN_SCOPED_RBAC=true`, `JOBS_RBAC_FORCE=true`), TTL without `X-Confirm` returns a safe no-op `{affected:0}` when a `domain` is provided.
> - Canonical endpoint for quarantine requeue is `POST /api/v1/jobs/batch/requeue_quarantined`; compatibility alias `POST /api/v1/jobs/batch/requeue-quarantined` remains supported.

## Configuration

- `JOBS_DB_URL` (optional): PostgreSQL DSN (e.g., `postgresql://user:pass@host:5432/db`). If not set, SQLite is used at `Databases/jobs.db`.
- `JOBS_LEASE_SECONDS` (default 60): Lease duration when acquiring.
- `JOBS_LEASE_RENEW_SECONDS` (default 30): Worker renewal cadence.
- `JOBS_LEASE_RENEW_JITTER_SECONDS` (default 5): Renewal jitter to avoid herds.
- `JOBS_LEASE_MAX_SECONDS` (default 3600): Cap for lease extension.
- `JOBS_ENFORCE_LEASE_ACK` (default true): Explicit override that forces lease enforcement on (takes precedence over the disable flag).
- `JOBS_DISABLE_LEASE_ENFORCEMENT` (default false): Compatibility switch that allows finalizing without `worker_id`/`lease_id`. Intended for legacy adapters and targeted tests; avoid enabling in production.
- `JOBS_ALLOWED_QUEUES` / `JOBS_ALLOWED_QUEUES_<DOMAIN>`: Comma-separated allowlists to restrict queue names (in addition to standard queues).
- `JOBS_MAX_JSON_BYTES` (default 1048576): Max serialized bytes for `payload` and `result`.
- `JOBS_JSON_TRUNCATE` (default false): If true, truncate oversize `payload`/`result` to a small marker instead of rejecting.
- `JOBS_ARCHIVE_BEFORE_DELETE` (default false): When true, `prune_jobs` copies rows to `jobs_archive` before deletion.
- `JOBS_SQLITE_SINGLE_UPDATE_ACQUIRE` (default false): Use an optional single-UPDATE acquisition path on SQLite under contention.
- `JOBS_ALLOWED_JOB_TYPES` / `JOBS_ALLOWED_JOB_TYPES_<DOMAIN>`: Comma-separated allowlists of job types. If set, `create_job` enforces membership.
- Exactly-once finalize (optional):
  - `JOBS_REQUIRE_COMPLETION_TOKEN` (default false): When true, workers should pass `completion_token` (e.g., the `lease_id`) to `complete_job`/`fail_job` to enforce idempotency.
  - `completion_token` is stored on finalize; repeated finalize with the same token becomes a no-op (returns True). A different token after finalization returns False.
- Audit bridge (optional):
  - `JOBS_AUDIT_ENABLED` (default false): Enable audit logging for job lifecycle events via `unified_audit_service`.
  - `JOBS_AUDIT_DB_PATH`: Optional override for the audit SQLite DB storing job events (`Databases/jobs_audit.db` by default).
  - `JOBS_AUDIT_RETENTION_DAYS`, `JOBS_AUDIT_BUFFER_SIZE`, `JOBS_AUDIT_FLUSH_SECONDS`: Tune retention and buffering for the audit bridge.
- Poison quarantine (optional):
  - `JOBS_QUARANTINE_THRESHOLD` (default 3): On repeated retryable failures with the same `error_code`, the job transitions to `quarantined` instead of re-queuing.
- Integrity sweeper (optional):
  - `JOBS_INTEGRITY_SWEEP_ENABLED` (default false), `JOBS_INTEGRITY_SWEEP_INTERVAL_SEC` (default 60), `JOBS_INTEGRITY_SWEEP_FIX` (default false).
  - Periodically flags (and optionally fixes) impossible states like leases on non-processing jobs and expired processing leases.
- Postgres RLS (optional):
  - `JOBS_PG_RLS_ENABLE` (default false): Enable row-level security policies that scope access to domains in `current_setting('app.domain_allowlist')`.
  - To scope a connection/session, set the allowlist before issuing queries/updates (example):
    - `SELECT set_config('app.domain_allowlist', 'chatbooks,prompt_studio', true);`
    - The policies will then allow access only to rows where `domain` is in that list.
- Metrics/Tracing buckets:
  - `JOBS_DURATION_BUCKETS`: CSV of float seconds for `duration_seconds` histogram buckets.
  - `JOBS_QUEUE_LATENCY_BUCKETS`: CSV of float seconds for `queue_latency_seconds` histogram buckets.
- Tracing & Events (optional):
  - `JOBS_TRACING` (default false): Log spans for job lifecycle events (create/acquire/complete/fail) with correlation metadata.
  - `JOBS_EVENTS_ENABLED` (default false): Emit event hooks for job state changes.
  - `JOBS_EVENTS_OUTBOX` (default false): Persist job events to an append-only outbox table `job_events` for CDC/streaming.
  - `JOBS_EVENTS_POLL_INTERVAL` (default 1.0): SSE poll interval for `/api/v1/jobs/events/stream`.
  - `JOBS_EVENTS_RATE_LIMIT_HZ` (default 0 = unlimited): Soft rate limit for event emission; excess writes are dropped.
  - Request/Trace correlation:
    - `X-Request-ID` is propagated from API → job row (request_id) when passed to `create_job` by endpoints (audio jobs wired; others can adopt).
    - A `trace_id` is generated per job when not provided; metrics can attach sampled exemplars with `JOBS_METRICS_EXEMPLARS=true` and `JOBS_METRICS_EXEMPLAR_SAMPLING` (default 0.01).
  - SLOs (owner/job_type): enable with `JOBS_SLO_ENABLE=true`, window `JOBS_SLO_WINDOW_HOURS` (default 6).
- TTL (optional):
  - `JOBS_TTL_ENFORCE` (default false): Enable periodic TTL sweeps in the metrics loop.
  - `JOBS_TTL_AGE_SECONDS`: Max age for queued jobs (by `created_at`).
  - `JOBS_TTL_RUNTIME_SECONDS`: Max runtime for processing jobs (by `COALESCE(started_at, acquired_at)`).
  - `JOBS_TTL_ACTION`: `cancel` (default) or `fail`.

### Queue Policies and Allowlists

- Allowed queues
  - Global: `JOBS_ALLOWED_QUEUES="q1,q2"`
  - Per-domain: `JOBS_ALLOWED_QUEUES_<DOMAIN>="q3,q4"` (e.g., `JOBS_ALLOWED_QUEUES_CHATBOOKS`)
  - Standard queues are always permitted: `default, high, low`.
- Allowed job types
  - Global: `JOBS_ALLOWED_JOB_TYPES="t1,t2"`
  - Per-domain: `JOBS_ALLOWED_JOB_TYPES_<DOMAIN>="t3,t4"`
- Queue controls (admin endpoints)
  - `POST /api/v1/jobs/queue/control` with `{ domain, queue, action: 'pause'|'resume'|'drain' }`
    - `pause`: block new acquisitions for the queue
    - `drain`: allow running jobs to finish, block new acquisitions
    - `resume`: clear both pause and drain
  - `GET /api/v1/jobs/queue/status?domain=...&queue=...` → `{ paused: bool, drain: bool }`

### Domain-Scoped RBAC (Admin)

- Flags:
  - `JOBS_DOMAIN_SCOPED_RBAC` (default false): Enforce domain scope for admin endpoints
  - `JOBS_REQUIRE_DOMAIN_FILTER` (default false): Require `domain` query/body field when RBAC is enabled
  - `JOBS_RBAC_FORCE` (default false): Apply checks even in single-user mode (useful for tests)
  - `JOBS_DOMAIN_ALLOWLIST_<USER_ID>`: Comma-separated domain allowlist for a specific user id
- TTL special-case: when RBAC is forced and a `domain` is provided, `POST /api/v1/jobs/ttl/sweep` without `X-Confirm` returns `{"affected": 0}` (no-op) instead of 400. This preserves guardrails while enabling RBAC-only validations.
- Prune & retention (optional):
  - config.txt [Jobs] keys map to env names without the `JOBS_` prefix (env overrides config).
  - `JOBS_PRUNE_ENFORCE` (default false): Run background prune sweeps (internal scheduler).
  - `JOBS_PRUNE_INTERVAL_SEC` (default 86400): Interval between prune sweeps (seconds).
  - `JOBS_PRUNE_DRY_RUN` (default false): Count candidates without deleting.
  - `JOBS_PRUNE_DOMAIN`, `JOBS_PRUNE_QUEUE`, `JOBS_PRUNE_JOB_TYPE`: Optional comma-separated scopes.
  - `JOBS_RETENTION_DAYS_TERMINAL`: Override all terminal states (`completed|failed|cancelled|quarantined`).
  - `JOBS_RETENTION_DAYS_COMPLETED` (default 30)
  - `JOBS_RETENTION_DAYS_FAILED` (default 60)
  - `JOBS_RETENTION_DAYS_CANCELLED` (default 60)
  - `JOBS_RETENTION_DAYS_QUARANTINED` (default 90)
  - `JOBS_RETENTION_DAYS_NONTERMINAL` (default 0): Days to retain non-terminal (`queued|processing`). Disabled when 0 (recommended unless intentional).
  - `JOBS_ARCHIVE_COMPRESS` (default false): When archiving, also write compressed copies of `payload`/`result` to `jobs_archive.payload_compressed` / `result_compressed`.
    - SQLite stores Base64-encoded `gzip64:<...>` strings. Postgres stores raw `BYTEA`.
  - `JOBS_ARCHIVE_COMPRESS_DROP_JSON` (default false): If true, set `payload`/`result` to NULL in the archive (use compressed columns only).
- Secret hygiene (creation):
  - `JOBS_SECRET_REJECT` (default false): Reject job creation if payload appears to contain secrets (keys or regex patterns).
  - `JOBS_SECRET_REDACT` (default false): Redact detected secrets with `***REDACTED***` (applies even if not rejecting).
  - `JOBS_SECRET_DENY_KEYS`: Comma-separated sensitive keys to flag (defaults include `api_key, authorization, password, token, secret, ...`).
  - `JOBS_SECRET_PATTERNS`: Semicolon-separated regex patterns to detect secrets (default includes OpenAI keys, AWS AKIA, GitHub PAT, JWT, Google API, Slack tokens).
- Graceful shutdown:
  - `JOBS_SHUTDOWN_WAIT_FOR_LEASES_SEC` (default 0): When >0, `/ready` flips to not ready and new acquisitions are gated; shutdown waits up to this many seconds for active leases to finish.
  - Readiness endpoints: `/ready` and `/health/ready` return `{"status": "not_ready"}` during shutdown to drain traffic.
  - Acquisition gate: on shutdown the server sets a global acquire gate so `acquire_next_job` returns `None` until restart.
- Testing time control:
   - `JOBS_TEST_NOW_EPOCH` (seconds since epoch): If set, JobManager’s internal clock uses this instant for lease renewals and TTL comparisons it controls, enabling time-travel tests without sleeps.
 - Domain-scoped RBAC (optional):
   - `JOBS_DOMAIN_SCOPED_RBAC` (default false): Enforce domain scope for admin endpoints.
   - `JOBS_REQUIRE_DOMAIN_FILTER` (default false): Require the `domain` query/body field for domain-scoped endpoints.
   - `JOBS_DOMAIN_ALLOWLIST_<USER_ID>`: Comma-separated domain allowlist for a specific user id.

Endpoint: `GET /api/v1/config/jobs` lists backend, flags, and standard queues.

## Core API (Python)

```python
from tldw_Server_API.app.core.Jobs.manager import JobManager

jm = JobManager()  # auto-selects SQLite or Postgres

# Create a job (queued)
job = jm.create_job(
    domain="chatbooks",
    queue="default",
    job_type="export",
    payload={"name": "Weekly", "chatbooks_job_id": "abc123"},
    owner_user_id="1",
    priority=5,
)

# Acquire next job with a lease
lease_seconds = 60
worker_id = "worker-1"
job = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=lease_seconds, worker_id=worker_id)
lease_id = job["lease_id"]

# Periodically renew (enforcement on)
jm.renew_job_lease(job["id"], seconds=lease_seconds, worker_id=worker_id, lease_id=lease_id)

# Complete with enforcement and exactly-once token (recommended)
jm.complete_job(job["id"], result={"path": "/path/to/file"}, worker_id=worker_id, lease_id=lease_id, completion_token=lease_id)

# Fail (retryable or terminal) with idempotent finalize token
jm.fail_job(job["id"], error="boom", retryable=True, worker_id=worker_id, lease_id=lease_id, completion_token=lease_id)

## Storage Architecture

- `JobManager` keeps its SQL inline and connects directly via sqlite3/psycopg. This retains the tuned acquisition logic, counters, and quarantine flows without an intermediate adapter.
- Schema includes supporting tables alongside `jobs`: `job_events` (outbox), `job_counters`, `job_queue_controls`, `job_sla_policies`, `job_attachments`, and optional `jobs_archive`. Dedicated tests cover each (`tests/Jobs/test_jobs_events_outbox_sqlite.py`, `test_jobs_admin_counters_sqlite.py`, `test_jobs_queue_controls_and_admin_sqlite.py`, `test_jobs_sla_gauges_sqlite.py`, `test_jobs_events_outbox_postgres.py`).
- Schema creation lives in `migrations.py` / `pg_migrations.py`; `_connect()` applies the migrations and pragmas on demand.
- When we add a dedicated storage wrapper it will reside at `app/core/Jobs/storage.py` and wrap the shared database adapters (`app/core/DB_Management/backends/`). Until then, the inline approach is the supported path.
- Both backends are exercised in the Jobs test suite (`tests/Jobs/test_jobs_manager_sqlite.py`, `tests/Jobs/test_jobs_manager_postgres.py`, `tests/Jobs/test_jobs_pg_concurrency_stress.py`), so direct SQL remains coverage-protected.

## Idempotency Scoping

- Idempotency is enforced per (domain, queue, job_type, idempotency_key).
  - Submitting a duplicate job with the same idempotency key and the same group returns the original row.
  - Reusing the same key in a different queue, job type, or domain creates a distinct job.
  - Postgres uses a composite unique index; SQLite mirrors this with a partial unique index (idempotency_key IS NOT NULL).

## Status Guardrails

- Legal transitions are enforced at the DB boundary:
  - `queued → processing → {completed, failed, cancelled}`
  - `processing → queued` only on retry (fail_job with retryable=True)
- `complete_job` and terminal `fail_job` affect rows only when `status='processing'`.
  - Completing/failing a non-processing job is a no-op (returns False, no change).
```

## Operational Guidance

- Leases and Reclaim:
  - Expired `processing` leases are reclaimed on the next acquire.
- Lease enforcement is enabled by default. Only use `JOBS_DISABLE_LEASE_ENFORCEMENT=1` for legacy adapters or targeted tests that cannot pass `worker_id`/`lease_id`.

- Priorities and Fairness:
  - Lower numeric `priority` means higher priority (default is 5).
  - Acquisition order is explicit and stable: `priority ASC`, then `available_at`/`created_at` ASC, then `id` ASC.

- Ready vs Scheduled semantics:
  - Ready = `status='queued'` and `available_at IS NULL OR available_at <= now()`
  - Scheduled = `status='queued'` and `available_at > now()`
  - Metrics and stats distinguish `queued` (ready) from `scheduled`; backlog = ready + scheduled.
  - Admin reschedule can move ready → scheduled (`/api/v1/jobs/batch/reschedule`), or scheduled → ready (`set_now=true`).

- Pruning:
  - Manager exposes `prune_jobs(statuses, older_than_days, domain=None, queue=None, job_type=None, dry_run=False)` to delete old `completed/failed/cancelled` jobs based on `completed_at` (fallback to `created_at`).
  - Use `domain`/`queue`/`job_type` to scope deletion; pass `dry_run=True` to preview the count only (no deletes).
  - Admin endpoint (admin-only): `POST /api/v1/jobs/prune`
    - Request (JSON):
      - `statuses`: list of statuses, e.g. `["completed","failed","cancelled"]`
      - `older_than_days`: integer days threshold (min 1)
      - Optional filters: `domain`, `queue`, `job_type`
      - Optional `dry_run`: boolean (default false)
      - Optional `detail_top_k`: integer (0..100). When `dry_run=true`, compute top-K groups by count (for preview).
    - Response: `{ "deleted": <int> }` - for dry runs this is the would-delete count.
  - Examples:
    - Preview prune for a single queue:
      ```bash
      curl -X POST "$BASE/api/v1/jobs/prune" \
        -H "X-API-KEY: $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
              "statuses": ["completed","failed"],
              "older_than_days": 14,
              "domain": "chatbooks",
              "queue": "default",
              "job_type": "export",
              "dry_run": true
            }'
      ```
    - Execute the prune (remove `dry_run` or set false):
      ```bash
      # Same body with "dry_run": false
      ```
  - WebUI:
    - Admin → Jobs includes a Prune panel with a “Dry Run (count only)” toggle and a “Saved Filters” badge reflecting current Domain/Queue/Job Type filters. Use Reset Filters to clear.

- Metrics:
  - Gauges: `jobs.queued{...}` (ready only), `jobs.scheduled{...}`, `jobs.processing{...}`, `jobs.backlog{...}` (ready + scheduled).
  - Histograms: `jobs.duration_seconds{...}`, `jobs.queue_latency_seconds{...}`, `jobs.retry_after_seconds{...}`.
  - Counters: `jobs.created_total{...}`, `jobs.completed_total{...}`, `jobs.cancelled_total{...}`, `jobs.retries_total{...}`, `jobs.failures_total{...,reason}`, `jobs.failures_by_code_total{...,error_code}`.
  - Lease tuning: `jobs.time_to_expiry_seconds{...}` histogram reflects remaining time on active leases.
  - Per-owner SLO gauges: queue latency and duration P50/P90/P99 per `{domain,queue,job_type,owner_user_id}`:
    - `jobs.queue_latency_p50_seconds` / `_p90_` / `_p99_`
    - `jobs.duration_p50_seconds` / `_p90_` / `_p99_`
  - Structured failure timeline stored on job rows: `failure_timeline` JSON (last ~10 entries) with `{ts, error_code, retry_backoff}` for WebUI analytics.

- Counters vs Reconcile:
  - `JOBS_COUNTERS_ENABLED` (default false): enable per-group counters in `job_counters` to avoid frequent COUNT(*) scans
  - Inline transitions update counters when fully scoped (e.g., create, acquire, finalize, TTL, batch ops)
  - Gauges use counters when available; otherwise they compute fresh counts
  - `JOBS_GAUGES_DEBOUNCE_MS` (default 0): debounce gauge updates in high-churn paths
  - Background reconcile (optional):
    - `JOBS_METRICS_GAUGES_ENABLED=true` emits SLO gauges
    - `JOBS_METRICS_RECONCILE_ENABLE=true` enables periodic reconcile of counters/gauges
    - `JOBS_METRICS_RECONCILE_GROUPS_PER_TICK` (default 100) caps groups per loop to avoid heavy scans

### Docker Compose (Postgres)

- The repository ships a `docker-compose.yml` with a `postgres` service. To run Jobs on Postgres when using Compose:
  - Set the DSN using the `postgres` service hostname inside the Compose network:
    - `export JOBS_DB_URL=postgresql://tldw_user:TestPassword123!@postgres:5432/tldw_users`
  - Start services:
    - `docker compose up --build`
  - From your host, you can also connect via the published port:
    - `export JOBS_DB_URL=postgresql://tldw_user:TestPassword123!@localhost:5432/tldw_users`
  - The Jobs manager will auto-provision the schema on first use.

### Running Postgres Jobs tests

- Ensure a Postgres instance is available (e.g., via Compose above) and set one of:
  - `export JOBS_DB_URL=postgresql://tldw_user:TestPassword123!@localhost:5432/tldw_users`
  - or `export POSTGRES_TEST_DSN=postgresql://...`
- Run only PG-marked Jobs tests:
  - `python -m pytest -m "pg_jobs" -v tldw_Server_API/tests/Jobs`

- TTL Sweep (optional):
  - Admin endpoint (admin-only): `POST /api/v1/jobs/ttl/sweep`
    - Request: `{ age_seconds?: int, runtime_seconds?: int, action: 'cancel'|'fail', domain?: string, queue?: string, job_type?: string }`
    - Action applies to queued jobs older than `age_seconds` and processing jobs running longer than `runtime_seconds`.
    - Response: `{ "affected": <int> }`.
  - Metrics integrate with the centralized metrics manager; no external setup required.

## Notes

- SQLite is suitable for single-instance or dev; Postgres is recommended for multi-worker deployments.
- Leader election for maintenance (Postgres): TTL and prune sweeps acquire per-`{domain,queue}` advisory locks via `pg_try_advisory_lock` so only one instance performs the sweep for that shard.
- Avoid long leases; prefer short leases with renewals.

## Developer Ergonomics

- Worker SDK
  - A lightweight helper lives at `tldw_Server_API/app/core/Jobs/worker_sdk.py`.
  - Provides auto-renew with jitter, optional progress heartbeats, and simple cancellation checks.
  - WorkerConfig keys:
    - `domain`, `queue`, `worker_id`
    - `lease_seconds` (default 30), `renew_threshold_seconds` (default 10), `renew_jitter_seconds` (default 5)
    - `backoff_base_seconds` (default 2), `backoff_max_seconds` (default 30)
    - `retry_on_exception` (default true), `retry_backoff_seconds` (default 10)
  - Example usage:
    ```python
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerSDK, WorkerConfig
    import asyncio

    async def handler(job):
        # do work...
        return {"ok": True}

    async def main():
        jm = JobManager()
        sdk = WorkerSDK(jm, WorkerConfig(domain="prompt_studio", queue="default", worker_id="w1"))
        await sdk.run(handler=handler)

    asyncio.run(main())
    ```

- Local CLI (prints cURL)
  - `Helper_Scripts/tldw_jobs.py` generates auth-aware cURL for stats, list, prune, TTL, and archive meta.
  - Examples:
    - `python Helper_Scripts/tldw_jobs.py stats --domain prompt_studio`
    - `python Helper_Scripts/tldw_jobs.py prune --domain prompt_studio --queue default --older_than_days 30 --dry_run`
    - `python Helper_Scripts/tldw_jobs.py archive-meta --job_id 123`

- Archive compression metadata
  - Admin endpoint: `GET /api/v1/jobs/archive/meta?job_id=<id>`
  - Returns booleans for JSON presence and compressed payload/result presence in `jobs_archive`.
- Workers should check `cancel_requested_at` and verify lease validity between chunks before writing final results.

## API: Stats and Listing

- `GET /api/v1/jobs/stats` (admin-only): returns aggregated counts per `(domain, queue, job_type)` with fields:
  - `queued` (ready only), `scheduled` (available in the future), `processing`, `quarantined`
  - Filterable by `domain`, `queue`, `job_type`

- `GET /api/v1/jobs/list` (admin-only): paged job listing with `domain`, `queue`, `status`, `owner_user_id`, `job_type`, `limit`
  - Sorting: `sort_by` in {`created_at`, `priority`, `status`} and `sort_order` in {`asc`, `desc`}.
  - Example: `GET /api/v1/jobs/list?domain=chatbooks&sort_by=priority&sort_order=asc`

## Admin Endpoint Safety

- Destructive admin endpoints require an explicit confirmation header to avoid accidental deletion:
  - `POST /api/v1/jobs/prune`: Set `X-Confirm: true` unless `dry_run: true`.
  - `POST /api/v1/jobs/ttl/sweep`: Set `X-Confirm: true`.
  - `POST /api/v1/jobs/batch/cancel` and `/jobs/batch/reschedule`: Set `X-Confirm: true` unless `dry_run: true`.
  - Without the header these endpoints return HTTP 400.

## Schema Notes

- New installs enforce:
  - CHECK status ∈ {queued, processing, completed, failed, cancelled, quarantined}
  - CHECK priority in [1..10], max_retries in [0..100]
  - progress_percent [0..100] and progress_message fields (workers can update via `renew_job_lease`)
  - Postgres adds partial indexes accelerating counts and acquisitions

- Forward migrations (Postgres):
  - On startup, `ensure_jobs_tables_pg()` creates tables and performs safe, idempotent forward migrations for missing columns used by the Jobs module (e.g., `completion_token`, `failure_streak_code/count`, `quarantined_at`, `progress_percent/message`, `error_code/class/stack`).
  - It also creates hot-path indexes concurrently, including the acquisition-order index: `(priority, COALESCE(available_at, created_at), id) WHERE status='queued'`.
  - This behavior is intended for dev/test and simple upgrades. For production change control, consider gating with an environment flag and running DDL as part of a managed migration process.
### Integrity Sweep (optional)

- Admin endpoint (admin-only): `POST /api/v1/jobs/integrity/sweep`
  - Request: `{ fix: boolean, domain?: string, queue?: string, job_type?: string }`

### Events (Outbox)

- Poll events:
  - `GET /api/v1/jobs/events?after_id=<cursor>&limit=<N>&domain=&queue=&job_type=` (admin-only)
- SSE stream:
  - `GET /api/v1/jobs/events/stream?after_id=<cursor>` (admin-only)
  - Emits text/event-stream with incremental IDs; clients can resume by passing `after_id`.
  - Requires `JOBS_EVENTS_OUTBOX=true` to persist events; otherwise events are process-local only.
  - Admin endpoints set per-request Postgres RLS context automatically when enabled.
  - Response: `{ non_processing_with_lease: int, processing_expired: int, fixed: int }`
  - When `fix=true`, clears stale lease fields on non-processing rows, and re-queues expired processing rows.

### Quarantine Triage (Admin Runbook)

- When jobs repeatedly fail with the same `error_code`, they transition to `quarantined` after `JOBS_QUARANTINE_THRESHOLD` retryable failures. Quarantine prevents automatic re-queueing to stop hot loops.
- Triage steps:
  - Inspect top failure codes and affected domains/queues in logs and metrics (`jobs.failures_by_code_total`).
  - Use `GET /api/v1/jobs/stats` to see `quarantined` counts by domain/queue/job_type.
  - Use `POST /api/v1/jobs/batch/requeue_quarantined` with `dry_run=true` to preview impact. Scope by `domain`, optionally `queue` and `job_type`.
  - Requeue with confirmation header when root cause is mitigated (e.g., fixed inputs, raised limits):
    - Header: `X-Confirm: true`
    - Example cURL (dry run):
      ```bash
      curl -X POST "$BASE/api/v1/jobs/batch/requeue_quarantined" \
        -H "X-API-KEY: $API_KEY" \
        -H "Content-Type: application/json" \
        -d '{
              "domain": "chatbooks",
              "queue": "default",
              "job_type": "export",
              "dry_run": true
            }'
      ```
    - Example cURL (real run with confirm):
      ```bash
      curl -X POST "$BASE/api/v1/jobs/batch/requeue_quarantined" \
        -H "X-API-KEY: $API_KEY" \
        -H "X-Confirm: true" \
        -H "Content-Type: application/json" \
        -d '{
              "domain": "chatbooks",
              "queue": "default",
              "job_type": "export",
              "dry_run": false
            }'
      ```
  - Consider raising or lowering `JOBS_QUARANTINE_THRESHOLD` per environment. Keep it conservative in production to guard against hot loops.
  - If a subset remains problematic, continue quarantine and investigate upstream data, credentials, or provider limits.
- Domain quotas (optional):
  - `JOBS_QUOTA_MAX_QUEUED` / `JOBS_QUOTA_MAX_QUEUED_<DOMAIN>` / `JOBS_QUOTA_MAX_QUEUED_USER_<USER_ID>` / `JOBS_QUOTA_MAX_QUEUED_<DOMAIN>_USER_<USER_ID>`
  - `JOBS_QUOTA_SUBMITS_PER_MIN` / `..._<DOMAIN>` / `..._USER_<USER_ID>` / `..._<DOMAIN>_USER_<USER_ID>`
  - `JOBS_QUOTA_MAX_INFLIGHT` / `..._<DOMAIN>` / `..._USER_<USER_ID>` / `..._<DOMAIN>_USER_<USER_ID>`
  - Precedence: domain+user > user > domain > global.
  - Notes: If `owner_user_id` is not supplied, inflight checks are skipped; configure workers to pass `owner_user_id` when needed. Inflight counts only `processing` jobs with a live lease (`leased_until` in the future); expired leases do not count.

### Encryption & Rotation

- Enabling encryption
  - Set `WORKFLOWS_ARTIFACT_ENC_KEY` to a strict base64-encoded 32-byte AES-256 key
  - Enable globally with `JOBS_ENCRYPT=true` or per-domain with `JOBS_ENCRYPT_<DOMAIN>=true`
  - When enabled, `payload`/`result` are stored as envelopes: `{ "_encrypted": {"_enc":"aesgcm:v1", ... } }`
- Reading with dual keys (rotation window)
  - Set `JOBS_CRYPTO_SECONDARY_KEY` to the previous key to allow reads during rotation
- Rotating stored rows (admin)
  - `POST /api/v1/jobs/crypto/rotate` (admin-only)
    - Body: `{ old_key_b64, new_key_b64, domain?, queue?, job_type?, fields?: ["payload","result"], limit?: 1000, dry_run?: true }`
    - Dry run counts candidates; execution requires `X-Confirm: true` header
- After rotation
  - Update `WORKFLOWS_ARTIFACT_ENC_KEY` to the new key and remove `JOBS_CRYPTO_SECONDARY_KEY`

### Deterministic Clock (Testing)

- `JOBS_TEST_NOW_EPOCH` can freeze time (UTC epoch seconds) for deterministic tests
- Internals (Postgres/SQLite) plumb this clock through acquisition, renew, TTL, and batch operations to keep behavior reproducible

### Signed Webhooks (optional)

- Enable a background worker that posts HMAC-signed webhooks for `job.completed` and `job.failed` using the job events outbox.
- Flags:
  - `JOBS_EVENTS_OUTBOX=true` (required on the webhook worker and every API/worker process that can mutate Jobs; webhook startup refuses to run without it)
  - `JOBS_WEBHOOKS_ENABLED=true`
  - `JOBS_WEBHOOKS_URL=https://example.com/jobs/webhook`
  - `JOBS_WEBHOOKS_SECRET_KEYS=primary,oldkey` (rotating; first key used to sign)
  - `JOBS_WEBHOOKS_INTERVAL_SEC` (default 1.0), `JOBS_WEBHOOKS_TIMEOUT_SEC` (default 5)
  - `JOBS_WEBHOOKS_CURSOR_PATH` (optional): path to persist the last delivered outbox id across restarts
- Headers sent:
  - `X-Jobs-Event`: `job.completed` | `job.failed`
  - `X-Jobs-Event-Id`: outbox id (monotonic cursor)
  - `X-Jobs-Timestamp`: epoch seconds
  - `X-Jobs-Signature`: `v1=<hex>` where `hex = HMAC_SHA256(secret, f"{ts}.{body}")`
- Body: JSON `{ event, attrs, job: {id,domain,queue,job_type}, created_at }`
- Verification example (Python):
  ```python
  import hmac, hashlib, time, json
  def verify(ts: str, body: bytes, sig_header: str, secrets: list[str], max_skew: int = 300) -> bool:
      if abs(int(time.time()) - int(ts)) > max_skew:
          return False
      try:
          scheme, value = sig_header.split('=', 1)
      except Exception:
          return False
      if scheme != 'v1':
          return False
      msg = f"{ts}.".encode() + body
      for sk in secrets:
          calc = hmac.new(sk.encode(), msg, hashlib.sha256).hexdigest()
          if hmac.compare_digest(calc, value):
              return True
      return False
  ```
