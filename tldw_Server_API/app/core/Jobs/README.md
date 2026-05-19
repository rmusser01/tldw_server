# Jobs

## 1. Descriptive of Current Feature Set

- Purpose: Durable background work for audio processing, embeddings, prompt studio tasks, file artifacts exports, and admin workflows. Supports SQLite/Postgres backends with leasing, retries, quarantine, quotas, metrics, and admin controls.
- Capabilities:
  - Create/list/inspect jobs; queue controls (pause/resume/drain);
  - Reschedule and retry-now actions; prune jobs by status/age with optional confirmation header.
  - Domain/queue/job_type scoping; per-owner quotas and optional RLS in Postgres.
  - Structured audit hooks for admin actions; metrics exported for Dashboarding.
- Inputs/Outputs:
  - Input: `create_job(domain, queue, job_type, payload, owner_user_id, ...)` and admin control requests.
  - Output: job rows with status transitions and timestamps; summaries and gauges for admin.
- Related Endpoints
  - Audio jobs submit/status/admin:
    - POST `/api/v1/audio/jobs/submit` — tldw_Server_API/app/api/v1/endpoints/audio_jobs.py:67
    - GET `/api/v1/audio/jobs/{job_id}` — tldw_Server_API/app/api/v1/endpoints/audio_jobs.py:149
    - GET `/api/v1/audio/jobs/admin/list` — tldw_Server_API/app/api/v1/endpoints/audio_jobs.py:190
    - GET `/api/v1/audio/jobs/admin/summary` — tldw_Server_API/app/api/v1/endpoints/audio_jobs.py:251
    - GET `/api/v1/audio/jobs/admin/summary-by-owner` — tldw_Server_API/app/api/v1/endpoints/audio_jobs.py:310
    - GET `/api/v1/audio/jobs/admin/owner/{owner_user_id}/processing` — tldw_Server_API/app/api/v1/endpoints/audio_jobs.py:351
    - GET/PUT `/api/v1/audio/jobs/admin/tiers/{user_id}` — tldw_Server_API/app/api/v1/endpoints/audio_jobs.py:422, 447
  - Jobs admin (generic):
    - POST `/api/v1/jobs/prune` — tldw_Server_API/app/api/v1/endpoints/jobs_admin.py:216
    - POST `/api/v1/jobs/queue/control` — tldw_Server_API/app/api/v1/endpoints/jobs_admin.py:280
    - GET `/api/v1/jobs/queue/status` — tldw_Server_API/app/api/v1/endpoints/jobs_admin.py:295
    - POST `/api/v1/jobs/reschedule` — tldw_Server_API/app/api/v1/endpoints/jobs_admin.py:322
    - POST `/api/v1/jobs/retry-now` — tldw_Server_API/app/api/v1/endpoints/jobs_admin.py:337
  - Embeddings re-embed scheduling (creates jobs):
    - tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:3391
  - File artifacts async exports:
    - POST `/api/v1/files/create` (async exports return `job_id`; status via GET `/api/v1/files/{file_id}`)
    - Worker: `tldw_Server_API/app/core/File_Artifacts/jobs_worker.py`

## 2. Technical Details of Features

- Architecture & Data Flow
  - `JobManager` manages job creation, leasing, retries, quarantine, quotas, and metrics: tldw_Server_API/app/core/Jobs/manager.py:1
  - Backends: SQLite (default) and Postgres (RLS and advisory locks supported in tests/migrations).
  - Admin controls operate via `JobManager` helpers to mutate flags and reschedule rows.
  - Event stream helpers provide pub/sub semantics to workers/clients; metrics integrate with the Metrics registry.

- Key Classes/Functions
  - `JobManager.create_job(...)` — entry to enqueue work; used by endpoints across modules (audio, embeddings, connectors, prompt studio).
  - `JobManager.set_queue_control(...)`, `JobManager._get_queue_flags(...)` — pause/resume/drain and status inspection.
  - `JobManager.reschedule_jobs(...)`, `JobManager.retry_now(...)`, `JobManager.prune_jobs(...)` — maintenance operations.

- Configuration
  - `JOBS_DB_URL` — if starts with `postgres`, enables Postgres backend; otherwise SQLite path or in-memory.
  - `JOBS_REQUIRE_CONFIRM` — require `X-Confirm: true` header for destructive prune unless `dry_run` or `TEST_MODE`.
  - `JOBS_UPDATE_GAUGES_ON_PRUNE` — refresh gauges after prune when fully scoped.

- Security
  - Admin endpoints use claim-first authorization (`get_auth_principal` + `RequireRole("admin")` / `RequirePermission(...)`). For Postgres, RLS is set per-user and per-domain before mutating rows.

## 3. Developer-Related/Relevant Information for Contributors

- Folder Structure
  - `manager.py` — core job operations; `metrics.py` — gauges/counters; `migrations*.py` — schema evolution; `event_stream.py` — lightweight streaming.
- Extension Points
  - Define job types per domain and implement workers using `worker_sdk.py`. Keep payloads JSON-serializable.
- Tests (selection)
  - Request/trace propagation into jobs (audio): tldw_Server_API/tests/Logging/test_trace_context.py:39–76
  - SQLite/Postgres stats and RBAC coverage: tldw_Server_API/tests/Jobs/test_jobs_stats_sqlite.py:34, tldw_Server_API/tests/Jobs/test_jobs_stats_postgres.py:44, tldw_Server_API/tests/Jobs/test_jobs_rbac_sqlite.py:60
  - Quotas/enforcement: tldw_Server_API/tests/Jobs/test_jobs_quotas_sqlite.py:11
  - Embeddings job path: tldw_Server_API/tests/e2e/test_embeddings_e2e.py:1
- Local Dev Tips
  - Start a Postgres via docker-compose to exercise RLS; set `JOBS_DB_URL` accordingly.
  - Use admin endpoints to pause/resume queues and to retry/reschedule stuck jobs during testing.
  - File artifacts exports: set `FILES_JOBS_WORKER_ENABLED=true` for in-process worker or run `python -m tldw_Server_API.app.core.File_Artifacts.jobs_worker` externally.
