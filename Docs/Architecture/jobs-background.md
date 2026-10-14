# Jobs and background work

This document describes the server's durable work substrate: the `JobManager` jobs table with leases and fair-share scheduling, the in-process worker pollers and the sidecar worker mode, the second internal `Scheduler` queue, the Jobs-vs-Scheduler decision rule, and the scheduling/automations/reminders/notifications services that ride on top.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Job manager | `app/core/Jobs/manager.py` | `JobManager` — `create_job`, `acquire_next_job(s)`, `renew_job_lease`, `complete_job` / `fail_job` / `cancel_job`, `prune_jobs`, `_recover_expired_processing_jobs`, the shutdown acquire gate |
| Job model | `app/core/Jobs/models.py` | `JobRecord`; terminal statuses `{completed, failed, cancelled, quarantined}` |
| Schema | `Jobs/migrations.py`, `pg_migrations.py` | SQLite (default `<project_root>/Databases/jobs.db`) and PG schemas, counters, DSN negotiation |
| Worker SDK | `Jobs/worker_sdk.py` | `WorkerSDK`, `WorkerConfig` (lease 30 s default), lease-evidence contexts |
| Fair share | `Jobs/fair_share.py` | owner/domain fair-share scoring mapped to priority |
| Events/telemetry | `Jobs/event_stream.py`, `metrics.py`, `audit_bridge.py` | outbox + SSE events, gauges, audit hooks |
| Admin surface | `app/api/v1/endpoints/jobs_admin.py` | ~25 routes: queue control, reschedule, retry-now, SLA policies, events SSE, stats, archive, batch ops |
| Deps | `app/api/v1/API_Deps/jobs_deps.py` | `get_job_manager()` — LRU-cached per jobs DB |
| Worker lifecycle | `app/services/lifecycle_workers.py`, `startup_*_jobs_pollers.py` | `ManagedWorker` registry, graceful stop, poller bootstrap |
| Scheduling services | `services/reminders_scheduler.py` + `reminder_jobs_worker.py`, `scheduled_task_automation_scheduler.py`, `scheduled_task_recurring_question_scheduler.py` | APScheduler feeds enqueuing Jobs (see below) |
| Agent tasks | `app/core/Scheduled_Tasks/agent_task_jobs.py` | `AUTOMATION_JOB_TYPE = "agent_task_run"`, executor registry |
| Internal Scheduler | `app/core/Scheduler/` | second queue system: `@task` handlers, leases + reaper, leader election, idempotency, dependencies; SQLite/PG/memory backends |
| Sync API | `app/api/v1/endpoints/sync.py` | durable server-side sync (v2 capabilities, conflicts, blob leases) — the chatbook Scheduling counterpart |
| Notifications | `core/Notifications/service.py`, `services/jobs_notifications_service.py` | email channel; job completed/failed → user-notification bridge |

## The job model

One generic `jobs` table (SQLite or PG — PG iff `JOBS_DB_URL` starts with `postgres`): domain, queue, job_type, owner/project ids, payload/result, status, priority (1–10, lower first), retries, `available_at`, lease fields (`leased_until`, `lease_id`, `worker_id`), error/cancel fields, idempotency key, completion token, `expired_lease_policy`. Standard queues `default|high|low` plus per-domain allowed queues (e.g. `reading: reading-digest`, `scheduled_tasks: scheduled-tasks`, `admin_webhooks: delivery`); extension via `JOBS_ALLOWED_QUEUES[_DOMAIN]`.

## Execution models

- **In-process pollers**: asyncio loops acquire jobs (poll ~1 s, lease default 120 s), process, and ack with worker+lease+completion-token. Managed by the lifecycle registry; app shutdown closes the acquire gate first so acquisitions return None, then quiesces pollers.
- **Sidecar mode**: `TLDW_WORKERS_SIDECAR_MODE=true` makes the API skip in-process pollers; `start-sidecars.sh` launches worker processes from `Docs/Deployment/sidecar_workers_manifest.json` (chatbooks, files, data_tables, prompt_studio, privilege_snapshots, audio, media_ingest, reading_import/digest, evals_abtest, optional connectors).
- **PG concurrency**: single-UPDATE `SKIP LOCKED` acquisition when enabled, else two-step pick+update; expired-lease recovery batched with `FOR UPDATE SKIP LOCKED` and a per-job policy (`consume_retry` vs `requeue_no_attempt`); advisory locks guard the dependency graph. Lease enforcement on ack is on by default (`JOBS_DISABLE_LEASE_ENFORCEMENT` to disable).

## Jobs vs Scheduler (the decision rule)

From AGENTS.md / ADR-003: **Jobs** for user-visible work, admin controls, and anything using the WorkerSDK; **the internal Scheduler** for internal orchestration via `@task` handlers. Recurring schedules use APScheduler enqueueing into either. Example: the workflows scheduler (APScheduler, `WORKFLOWS_SCHEDULER_ENABLED`) enqueues `workflow_run` tasks into the core Scheduler whose handlers live in `core/Scheduler/handlers/` (workflows, watchlists, ACP), persisting in the scheduler DB (default `<project_root>/Databases/scheduler.db`).

## Scheduling services (dataflow)

1. **Reminders**: task CRUD at `/api/v1/tasks` → the APScheduler reminders feed (rescan loop, misfire-safe) → due tasks enqueue `{domain: notifications, job_type: reminder_due}` → the reminder worker executes → notification row + optional email. Ships enabled; timezone and rescan configurable (rescan floor 30 s).
2. **Automations (scheduled agent tasks)**: definitions CRUD + capability checks at `/api/v1/scheduled-tasks/*` → the automation scheduler's arming model (per-occurrence date triggers, misfire-safe re-arm) → `agent_task_run` jobs → the executor registry; gated by deployment certification; the scheduler **ships OFF**. This is the server side of the chatbook's server-offloaded scheduled tasks (ADR-077 in the chatbook repo).
3. **Sync**: `/api/v1/sync/*` durable mutation batches, conflicts, and blob leases (ADR-034) — consumed by the chatbook's Sync v2 client.
4. **Job→notification bridge**: job completed/failed events become user notifications (single-user default user id configurable).

## Config keys (selection; exact)

`JOBS_DB_URL`, `JOBS_DB_PATH`, `JOBS_LEASE_SECONDS` (120), adaptive lease keys (`JOBS_ADAPTIVE_LEASE_*`), `JOBS_POLL_INTERVAL_SECONDS` (1), `JOBS_EXPIRED_RECOVERY_BATCH_SIZE`, `JOBS_MAX_PER_USER` (5) / `JOBS_MAX_PER_ORG` (20), `JOBS_QUARANTINE_THRESHOLD` (2), `JOBS_ARCHIVE_*`, `JOBS_ENCRYPT` + secret-pattern deny keys, `JOBS_ALLOWED_JOB_TYPES` / `JOBS_ALLOWED_QUEUES[_DOMAIN]`, `JOBS_REQUIRE_COMPLETION_TOKEN`, `JOBS_PG_SINGLE_UPDATE_ACQUIRE`, `JOBS_DISABLE_LEASE_ENFORCEMENT`, `SCHEDULER_DATABASE_URL` / `WORKFLOWS_SCHEDULER_DATABASE_URL`, `SCHEDULER_DB_POOL_MIN/MAX`, `REMINDERS_SCHEDULER_ENABLED/_TZ/_RESCAN_SEC`, `SCHEDULED_TASKS_AUTOMATION_SCHEDULER_ENABLED` (off), `AUTOMATION_JOBS_QUEUE`, `WORKFLOWS_SCHEDULER_ENABLED/_TZ`, `TLDW_WORKERS_SIDECAR_MODE`.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Lease expiry | Recovery batch requeues with retry consumed (or not, per policy); over max retries → failed |
| Unsupported job_type in a worker | `fail_job(retryable=False)` |
| Repeated failures | Quarantine at threshold; admin requeue path exists |
| WorkerSDK renewal lost | Typed terminalization conflict; detached completion handled |
| Shutdown | Acquire gate closes before poller quiesce — no half-acquired work |

## Verified gotchas

1. `Docs/Code_Documentation/Data_Flow_Atlas.md` presents Redis as a jobs backend — **Jobs is strictly SQLite/Postgres**; Redis appears only in the embeddings pipelines.
2. The two `Jobs_Manager.md` / `Jobs_Module.md` docs and the module READMEs are current (acquisition ordering, PG two-path behavior, defaults all verified).
3. `ADR-077`/`ADR-184` cited in the automation scheduler's comments live in the **chatbook** repo's ADR set, not this repo's `Docs/ADR/` (which ends at 067).
4. Domain-restricted queues are allowlists — adding a queue name without registering it in `JOBS_ALLOWED_QUEUES[_DOMAIN]` silently strands jobs.

## Governing decisions

ADR-003 (`Docs/ADR/003-jobs-vs-scheduler-default.md`), ADR-034 (`034-durable-server-origin-sync-mutation-batches.md`), ADR-041 (`041-scheduled-agent-execution-feasibility.md`), ADR-051 (retry backoff schedules), ADR-058 (jobs completion row identity). Module READMEs under `core/Jobs/` and `core/Scheduler/` are current.

## Related docs

- [app-shell.md](./app-shell.md) — lifespan and worker bootstrap
- [database-layer.md](./database-layer.md) — the store backends jobs DBs ride on
- [authnz.md](./authnz.md) — the admin BYOK/backup workers as pattern examples
