# Jobs

Jobs is the durable background-work backend for user-visible and admin-visible
work: audio jobs, embeddings stages, file exports, prompt-studio tasks, reading
imports/digests, model acquisition, VN assets, and other domain queues. It
supports SQLite and Postgres backends, leasing, retries, queue controls,
quotas, metrics, events, audit hooks, and worker SDK utilities.

## Start Here

- Core manager: `manager.py`.
- Data models and migrations: `models.py`, `migrations.py`, `pg_migrations.py`,
  and `pg_util.py`.
- Worker helpers: `worker_sdk.py`, `worker_utils.py`, and `fair_share.py`.
- Events/metrics/audit: `event_stream.py`, `metrics.py`, `audit_bridge.py`, and
  `tracing.py`.
- API dependencies/endpoints: `app/api/v1/API_Deps/jobs_deps.py` and
  `app/api/v1/endpoints/jobs_admin.py`.
- Tests: `tests/Jobs/`.

## Responsibilities

- Create jobs with domain, queue, type, owner, payload, idempotency, and quota
  metadata.
- Lease, renew, complete, fail, retry, reschedule, quarantine, and prune jobs.
- Pause, resume, drain, and summarize queues for admins.
- Emit structured events and metrics for workers, SSE, and dashboards.
- Support Postgres RLS/advisory-lock behavior where configured.

## Module Map

- `manager.py` is the main API for enqueueing and lifecycle transitions.
- `models.py` defines statuses and typed job shapes.
- `migrations.py` and `pg_migrations.py` create/upgrade SQLite and Postgres
  schemas.
- `worker_sdk.py` gives workers a consistent acquire/process/complete loop.
- `fair_share.py` schedules acquisition across owners/domains.
- `metrics.py` exports gauges/counters and reconciliation helpers.
- `event_stream.py` handles lightweight pub/sub event delivery.

## How It Connects

- Audio, embeddings, file artifacts, prompt studio, collections, VN assets,
  prototype workspaces, research, and other modules enqueue domain jobs.
- Admin endpoints use Jobs manager methods for queue controls and pruning.
- Billing/Usage/Resource Governance can influence quotas and cost accounting.
- Logging/Tracing add request IDs and trace metadata to job records.

## Architecture Notes

### Core Flow

- Domain modules create work through `JobManager.create_job(...)` with domain, queue, type, owner, payload, idempotency, quota, and scheduling metadata.
- Workers acquire jobs with a lease, process the payload, renew when needed, and finish through `complete_job` or `fail_job`. `WorkerSDK` wraps that loop so cancellation, structured errors, metrics, and retries stay consistent.
- Admin queue controls in `jobs_admin.py` pause, resume, drain, prune, retry, reschedule, cancel, and requeue quarantined jobs through the same manager and backend state.
- `event_stream.py`, `metrics.py`, `audit_bridge.py`, and `tracing.py` observe lifecycle transitions; they should not own domain-specific business logic.

### State And Data

- SQLite and Postgres schemas are maintained in `migrations.py`, `pg_migrations.py`, and `pg_util.py`; persisted fields need both backends and migration tests.
- Job payloads are durable JSON contracts across API processes, workers, and deployments. Keep payloads small, version-tolerant, and owned by the module that enqueues them.
- Idempotency is enforced with persisted keys, and completion/failure can use lease or completion tokens for repeated finalize attempts.

### Security And Operations

- Postgres paths preserve RLS and domain allowlist behavior through endpoint context and manager calls. Do not fabricate owner or principal context to make an admin route easier.
- Destructive admin operations such as prune, batch cancel, and quarantine requeue require scoped filters and confirmation behavior outside test mode.
- Leases are the concurrency boundary. Workers must complete, fail, or renew with the expected worker and lease identifiers rather than updating rows directly.

### Extension Checklist

- New domain job: define the payload near the owning module, enqueue through `JobManager`, add worker coverage, and include domain/queue owner tests.
- New persisted field: update SQLite migration, Postgres migration, model serialization, manager reads/writes, and both migration suites.
- New admin control: update `jobs_admin.py`, RBAC/RLS tests, queue-control tests, and audit or metrics expectations.

## Extension Points

- Define domain-specific job types next to the owning module, then enqueue via
  `JobManager.create_job(...)` with JSON-serializable payloads.
- Use `WorkerSDK` for new workers so lease renewal, structured errors, metrics,
  and cancellation behavior stay consistent.
- Add migrations in both SQLite and Postgres paths when changing persisted
  fields.

## Testing

- Core lifecycle and SQLite/Postgres behavior: `tests/Jobs/test_jobs_manager.py`,
  `tests/Jobs/test_jobs_manager_postgres.py`, and
  `tests/Jobs/test_jobs_batch_lifecycle_sqlite.py`.
- Admin controls and pruning: `tests/Jobs/test_jobs_admin_endpoints_sqlite.py`,
  `tests/Jobs/test_jobs_queue_controls_and_admin_sqlite.py`, and
  `tests/Jobs/test_jobs_prune_sqlite.py`.
- Quotas/RLS/fair share: `tests/Jobs/test_jobs_quotas_sqlite.py`,
  `tests/Jobs/test_jobs_rls_postgres.py`, and `tests/Jobs/test_fair_share.py`.
- Worker SDK and events: `tests/Jobs/test_worker_sdk.py`,
  `tests/Jobs/test_jobs_events_sqlite.py`, and `tests/Jobs/test_jobs_sse_smoke.py`.

## Gotchas

- Job payloads cross process and database boundaries; keep them small,
  JSON-serializable, and version-tolerant.
- Prune/reschedule/retry operations are admin-sensitive and often require
  confirmation headers outside test mode.
