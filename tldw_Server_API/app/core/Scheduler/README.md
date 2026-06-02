# Scheduler

Core task queue, worker pool, and execution orchestration. Provides an atomic, DB‑backed task system with idempotency, dependencies, leases, leader election, and multiple backends. The Workflows recurring scheduler uses APScheduler to enqueue `workflow_run` tasks into this core Scheduler.

## 1. Descriptive of Current Feature Set

- Task queue + workers
  - Submit single or batched tasks with priorities, dependencies, idempotency keys, metadata, and per‑task auth context.
  - Worker pool manages queue consumers; scale per‑queue workers at runtime.
- Backends
  - SQLite and PostgreSQL via unified backend factory; in‑memory utilities for tests exist.
  - Leader election ensures single‑node cleanup/monitor leadership in distributed runs.
- Safety & observability
  - Leases with reaper, safe write buffer with crash recovery, dependency validation (missing/cycles), and monitoring hooks.
  - Authorization checks for cancel/list operations via `TaskAuthorizer`.
- Integration
  - Workflows service enqueues `workflow_run` into queue `workflows`; Watchlists may enqueue `watchlist_run`.
- Related endpoints (scheduler control lives under the Workflows Scheduler API)
  - Scheduler is internal; public scheduling endpoints are exposed at `/api/v1/scheduler/workflows` (see below).

Related Endpoints (file:line)
- Workflows Scheduler API: tldw_Server_API/app/api/v1/endpoints/scheduler_workflows.py:18
  - Create schedule: 78
  - Admin rescan: 102
  - List schedules: 132
  - Get schedule: 163
  - Update schedule: 260
  - Delete schedule: 283
  - Run now: 303
  - Dry run (cron validation): 334

Related Services/DB
- Recurring scheduler service (APScheduler): tldw_Server_API/app/services/workflows_scheduler.py:1
- Per-user schedules DB: tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py:1
- Core Scheduler API (internal): tldw_Server_API/app/core/Scheduler/scheduler.py:1

## Architecture Notes

### Core Flow

- Internal callers submit work through `Scheduler.submit()` or `submit_batch()`. The scheduler validates idempotency, dependency existence, payload shape, and auth context before adding tasks to the safe write buffer.
- The write buffer persists tasks to the configured backend, then the worker pool leases ready tasks, invokes the registered handler from `base/registry.py`, updates status, and releases or reclaims leases.
- APScheduler-backed services such as Workflows recurring schedules and Watchlists enqueue task handlers (`workflow_run`, `watchlist_run`) into this core Scheduler; they do not execute workflow business logic themselves.
- `TaskAuthorizer` and `AuthContext` are part of the task metadata path and guard cancellation or administrative operations.

### State And Data

- Backend choice comes from `config.py` and `backends/`: SQLite and Postgres are durable, while memory backend support is for tests and local utilities.
- Leases, task status, dependency edges, idempotency keys, payloads, and auth metadata are backend state, not worker-local state.
- Leader election and the lease reaper coordinate cleanup and single-actor duties across processes.

### Security And Operations

- Scheduler is for internal orchestration with dependency handling. Use Jobs for new user-facing background work that needs admin controls, quotas, or queue visibility.
- Handlers must be idempotent because retries, lease expiry, or process restarts can re-run work after partial progress.
- Batch submission is the safer path when tasks reference each other because dependency and idempotency validation are evaluated before enqueueing the group.

### Extension Checklist

- New task handler: register it with `@task`, define payload validation, add handler tests, and add enqueueing tests in the owning module.
- New backend behavior: update the base backend contract, SQLite/Postgres implementations, and backend-specific tests.
- New scheduler consumer: document whether it should use Scheduler or Jobs, include idempotency keys, and test lease or retry behavior.

## 2. Technical Details of Features

- Architecture & components
  - `Scheduler`: orchestrates backends, write buffer, worker pool, leader election, and services (leases, dependencies, payloads).
    - File: tldw_Server_API/app/core/Scheduler/scheduler.py:1
  - Backends: `backends/{sqlite_backend,postgresql_backend,memory_backend}.py`; constructed via `create_backend` and `BackendManager`.
  - Registry: `base/registry.py` registers task handlers (e.g., `workflow_run`, `watchlist_run`).
- Data flow (submit)
  - `submit()` → prepare task (idempotency, dependencies, auth) → buffer add → backend persist → worker pickup → handler execution → status update.
  - `submit_batch()` validates all tasks atomically before enqueue.
- Concurrency & safety
  - Leases for running tasks with periodic reaper; write buffer flush, crash recovery on startup; leader election for single‑actor duties.
  - Dependency service validates existence and circular dependencies (best‑effort for buffered tasks).
- Authorization
  - `TaskAuthorizer` and `AuthContext` guard cancellations and admin operations; canceled tasks release leases.
- Configuration
  - `SchedulerConfig` via env and code: see `tldw_Server_API/app/core/Scheduler/config.py` for knobs (DB URL, write‑buffer sizes, lease TTLs, worker concurrency, etc.).
- Observability
  - `get_status()`, `get_queue_status()`, and metrics/monitoring helpers under `monitoring/` directory; HTTP metrics middleware aggregates at app level.

## 3. Developer‑Related/Relevant Information for Contributors

- Folder structure
  - `core/Scheduler/scheduler.py` — main orchestrator.
  - `core/Scheduler/base/` — core types, registry, exceptions, task model.
  - `core/Scheduler/backends/` — DB adapters and factory.
  - `core/Scheduler/core/` — worker pool, leader election, write buffer.
  - `core/Scheduler/services/` — lease, dependency, payload services.
- Patterns & tips
  - Register new task handlers via `@task` in `base/registry.py`; keep handlers side‑effect safe and idempotent where possible.
  - Use idempotency keys when enqueueing from external triggers; include `user_id`/`tenant_id` in `metadata` for auth and auditing.
  - Prefer batch submission for multiple related tasks to get atomic validation.
- Tests
  - Scheduler is exercised via Workflows Scheduler tests: tldw_Server_API/tests/Workflows/test_workflows_scheduler.py:51, 62, 85, 107, 124
  - Watchlists jitter behavior: tldw_Server_API/tests/Watchlists/test_watchlists_scheduler_jitter.py:35
  - AuthNZ virtual key rate limits: tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_limits_unit.py:13
- Pitfalls
  - Ensure the core scheduler is started (`get_global_scheduler()`) before enqueueing (services handle this in app lifespan).
  - Cron/timezone validation uses APScheduler; always pass IANA tz names (e.g., `UTC`).
  - PostgreSQL backends require proper connection strings and schema privileges; see DB_Management docs.
