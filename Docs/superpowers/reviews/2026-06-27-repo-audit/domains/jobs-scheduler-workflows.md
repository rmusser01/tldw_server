# Jobs, Scheduler, And Workflows Domain Review

## Scope

- Baseline: `origin/dev` at `669092178b0ba0fa1e840a37250b0deb55acd5a3`
- Report owner: Jobs, Scheduler, and Workflows
- In scope: queue semantics, retries, lifecycle, idempotency, worker behavior, workflow orchestration, task isolation, and related tests.
- Out of scope: remediation implementation and new job features.
- Reviewed worktree: `/Users/appledev/Documents/GitHub/tldw_server/.worktrees/comprehensive-repo-audit-2026-06-27`
- User constraints honored: report-only audit; no production/source edits; no Backlog task creation or updates; no dependency installs, services, Docker, or network access.

## Findings Table

| ID | Evidence Tier | Evidence Strength | Severity | Confidence | Category | Title | Status | Validation Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CANDIDATE-jobs-scheduler-workflows-001 | confirmed_issue | static_confirmed | medium | high | data_durability | Async workflow runs are handed to an in-process daemon-thread scheduler with no durable recovery for queued runs | open | needs_reproduction |
| CANDIDATE-jobs-scheduler-workflows-002 | likely_risk | static_confirmed | medium | medium | reliability | Recurring workflow and ACP schedule fires submit non-idempotent Scheduler tasks | open | needs_reproduction |

## Index Mapping

The coordinator should map these candidate IDs into the canonical audit index ID format if accepted. Required index fields are represented in each detailed finding below.

- `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/jobs-scheduler-workflows.md`
- `owner_domain`: `Jobs, Scheduler, and Workflows`

## Confirmed Issues

### CANDIDATE-jobs-scheduler-workflows-001: Async workflow runs are handed to an in-process daemon-thread scheduler with no durable recovery for queued runs

- `severity`: `medium`
- `confidence`: `high`
- `category`: `data_durability`
- `evidence_tier`: `confirmed_issue`
- `evidence_strength`: `static_confirmed`
- `status`: `open`
- `validation_status`: `needs_reproduction`
- `affected_paths`:
  - `tldw_Server_API/app/api/v1/endpoints/workflows.py`
  - `tldw_Server_API/app/core/Scheduler/handlers/workflows.py`
  - `tldw_Server_API/app/core/Workflows/engine.py`
  - `tldw_Server_API/app/services/workflows_db_maintenance.py`
- `evidence`:
  - HTTP ad-hoc workflow runs persist a `workflow_runs` row and then call `engine.submit(run_id, run_mode)`: `tldw_Server_API/app/api/v1/endpoints/workflows.py:2358-2379`.
  - The durable Scheduler `workflow_run` handler creates a fresh run row with `idempotency_key=None`, then returns after handing the run to `WorkflowEngine.submit`: `tldw_Server_API/app/core/Scheduler/handlers/workflows.py:90-125`.
  - `WorkflowEngine.submit` delegates async execution to `WorkflowScheduler.instance().schedule(...)`: `tldw_Server_API/app/core/Workflows/engine.py:1492-1498`.
  - `WorkflowScheduler` stores pending runs in an in-memory `deque`, tracks active runs in process memory, and starts execution in daemon threads: `tldw_Server_API/app/core/Workflows/engine.py:2193-2238`, `tldw_Server_API/app/core/Workflows/engine.py:2306-2316`.
  - The orphan reaper is invoked from `start_run` and covers stale running step rows, not queued workflow rows stranded before a daemon thread starts: `tldw_Server_API/app/core/Workflows/engine.py:657-660`, `tldw_Server_API/app/core/Workflows/engine.py:1842-1984`. The maintenance service inspected is database checkpoint/VACUUM oriented and does not repair queued workflow runs: `tldw_Server_API/app/services/workflows_db_maintenance.py`.
- `impact`: A process crash or shutdown after row creation but before or during daemon-thread execution can leave a workflow run queued/stale without a durable worker owning it. For scheduled workflows, the core Scheduler task can complete after enqueueing into the in-process workflow engine, so Scheduler success does not mean the workflow finished. Daemon threads are also terminated abruptly at process exit.
- `recommendation`: Execute workflow runs end-to-end inside a durable Jobs or Scheduler worker, or add a startup/periodic repair loop that finds queued/running workflow rows without active execution and requeues them idempotently. Avoid daemon threads for durable work, or make them a thin local executor backed by durable queue state and explicit shutdown/drain semantics.

## Likely Risks

### CANDIDATE-jobs-scheduler-workflows-002: Recurring workflow and ACP schedule fires submit non-idempotent Scheduler tasks

- `severity`: `medium`
- `confidence`: `medium`
- `category`: `reliability`
- `evidence_tier`: `likely_risk`
- `evidence_strength`: `static_confirmed`
- `status`: `open`
- `validation_status`: `needs_reproduction`
- `affected_paths`:
  - `tldw_Server_API/app/services/workflows_scheduler.py`
  - `tldw_Server_API/app/api/v1/endpoints/scheduler_workflows.py`
  - `tldw_Server_API/app/core/Scheduler/handlers/workflows.py`
  - `tldw_Server_API/app/core/Scheduler/scheduler.py`
  - `tldw_Server_API/app/core/Scheduler/backends/sqlite_backend.py`
  - `tldw_Server_API/app/core/Scheduler/backends/postgresql_backend.py`
  - `tldw_Server_API/tests/Workflows/test_workflows_scheduler.py`
- `evidence`:
  - The recurring workflow service starts a local `AsyncIOScheduler` in process and loads schedules into it: `tldw_Server_API/app/services/workflows_scheduler.py:124-154`.
  - ACP schedule fires call `core_scheduler.submit(...)` without an `idempotency_key`: `tldw_Server_API/app/services/workflows_scheduler.py:516-521`.
  - Workflow schedule fires also call `core_scheduler.submit(...)` without an `idempotency_key`: `tldw_Server_API/app/services/workflows_scheduler.py:610-616`.
  - The downstream `workflow_run` handler creates a new UUID run and explicitly stores `idempotency_key=None`: `tldw_Server_API/app/core/Scheduler/handlers/workflows.py:90-105`.
  - The core Scheduler already has idempotency support (`get_task_by_idempotency_key`) and the SQLite backend enforces a unique non-null idempotency key, but these scheduled submissions do not use it: `tldw_Server_API/app/core/Scheduler/scheduler.py:597-619`, `tldw_Server_API/app/core/Scheduler/backends/sqlite_backend.py:210-213`.
  - Existing workflow scheduler tests cover schedule firing, owner resolution, run-now behavior, and orphan-step requeue, but the inspected test slice does not exercise multi-process schedule ownership or duplicate schedule-fire dedupe: `tldw_Server_API/tests/Workflows/test_workflows_scheduler.py`, `tldw_Server_API/tests/Workflows/test_orphan_requeue_unit.py`.
- `impact`: In a multi-worker deployment, or during overlapping rescan/misfire/restart behavior, two APScheduler instances can submit the same logical schedule fire. Because the core Scheduler task and the created workflow run are not keyed by schedule ID plus fire window, duplicates can execute side effects and consume quota/cost more than once. Local `concurrency_mode` controls APScheduler `max_instances` inside one process only; it does not provide cross-process dedupe.
- `recommendation`: Add a deterministic idempotency key for each schedule fire, namespaced by handler, schedule ID, owner/user, and fire time window. Pass that key to `core_scheduler.submit` and propagate it into workflow run creation where appropriate. Alternatively or additionally, add a distributed leader/lease for the recurring APScheduler service. Add tests that submit the same logical schedule fire twice and assert one Scheduler task/workflow run.

## Improvement Opportunities

- No separate improvement candidates were promoted beyond the two reliability/data-durability findings above.
- Residual design note: core Scheduler idempotency keys are globally unique (`tasks.idempotency_key`), so any future user-provided Scheduler idempotency surface should namespace keys by tenant/user/handler to avoid accidental cross-user collisions.

## Coverage And Evidence

### Files Inspected

- `Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/endpoint-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/backend-test-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/db-migration-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt`
- `tldw_Server_API/app/core/Jobs/README.md`
- `tldw_Server_API/app/core/Jobs/manager.py`
- `tldw_Server_API/app/core/Jobs/models.py`
- `tldw_Server_API/app/core/Jobs/worker_sdk.py`
- `tldw_Server_API/app/core/Scheduler/README.md`
- `tldw_Server_API/app/core/Scheduler/scheduler.py`
- `tldw_Server_API/app/core/Scheduler/backends/sqlite_backend.py`
- `tldw_Server_API/app/core/Scheduler/backends/postgresql_backend.py`
- `tldw_Server_API/app/core/Scheduler/handlers/workflows.py`
- `tldw_Server_API/app/core/Workflows/README.md`
- `tldw_Server_API/app/core/Workflows/engine.py`
- `tldw_Server_API/app/core/DB_Management/Workflows_DB.py`
- `tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py`
- `tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py`
- `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py`
- `tldw_Server_API/app/api/v1/endpoints/scheduler_workflows.py`
- `tldw_Server_API/app/api/v1/endpoints/workflows.py`
- `tldw_Server_API/app/services/workflows_scheduler.py`
- `tldw_Server_API/app/services/workflows_db_maintenance.py`
- Related tests under `tldw_Server_API/tests/Jobs`, `tldw_Server_API/tests/Scheduler`, `tldw_Server_API/tests/Workflows`, `tldw_Server_API/tests/Chat_Workflows`, `tldw_Server_API/tests/AudioJobs`, and `tldw_Server_API/tests/Services` were inventoried from the provided audit evidence; selected workflow/job tests were run locally.

### Tests Or Scans Run

- `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workflows/test_workflows_scheduler.py tldw_Server_API/tests/Workflows/test_orphan_requeue_unit.py tldw_Server_API/tests/Jobs/test_worker_sdk.py -q`
  - Result: `27 passed, 325 warnings in 16.72s`.
- Existing audit evidence consumed:
  - `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt`
  - `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/backend-test-inventory.txt`
  - `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/db-migration-inventory.txt`
  - `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/endpoint-inventory.txt`

### Blocked Or Unverified Areas

- No environment-changing setup was requested or performed.
- No Docker, service startup, dependency installation, or network access was used.
- Multi-process APScheduler duplicate-fire behavior was not runtime-reproduced because that would require a coordinated multi-worker/service environment outside the permitted local static/test scope.
- Crash/shutdown recovery for queued workflow runs was not runtime-reproduced because that would require process-kill orchestration and potentially mutating runtime state; the finding is based on static source evidence.
- Bandit was not rerun for this report-only audit; the precomputed app summary was reviewed.

### Evidence Notes

- Jobs core coverage looked comparatively mature for admin gating, queue pause/resume/drain, lease enforcement, quotas, and WorkerSDK finalization semantics. No Jobs-specific candidate was promoted from the inspected static paths and selected passing tests.
- Existing workflow scheduler tests validate common single-process behavior, including schedule execution, owner attribution, run-now, and orphan-step requeue. The residual gaps are cross-process schedule ownership/dedupe and durable recovery of workflow rows queued into the in-process engine.
- Final git status showed unrelated changes outside this owned report, including `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/command-log.md` and untracked watchlist template files under `tldw_Server_API/Config_Files/templates/watchlists/`; this review did not edit those files.
