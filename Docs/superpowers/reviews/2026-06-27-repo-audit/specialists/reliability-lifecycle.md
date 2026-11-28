# Reliability And Async Lifecycle Specialist Review

## Scope

- Baseline: `origin/dev` at `669092178b0ba0fa1e840a37250b0deb55acd5a3`
- Report owner: Reliability and async lifecycle
- In scope: async shutdown, background workers, scheduler/jobs, retries, idempotency, DB transaction durability, resource cleanup, and reliability-relevant domain findings.
- Out of scope: remediation implementation and performance feature work.

## Findings Table

| ID | Evidence Tier | Evidence Strength | Severity | Confidence | Category | Title | Status | Validation Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUDIT-2026-06-27-REL-001 | likely_risk | static_confirmed | medium | high | reliability | Workflow continuation resumes are fire-and-forget tasks outside durable scheduler ownership | open | needs_reproduction |

## Index Mapping

New specialist finding details for index ingestion:

- `id`: `AUDIT-2026-06-27-REL-001`
- `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/reliability-lifecycle.md`
- `owner_domain`: `Reliability and async lifecycle`
- `affected_paths`: `tldw_Server_API/app/api/v1/endpoints/workflows.py`, `tldw_Server_API/app/api/v1/endpoints/research_runs.py`, `tldw_Server_API/app/core/Workflows/engine.py`, `tldw_Server_API/app/core/Workflows/research_wait_bridge.py`, `tldw_Server_API/tests/Workflows/test_workflows_api.py`, `tldw_Server_API/tests/Workflows/test_versions_idempotency.py`, `tldw_Server_API/tests/Workflows/test_orphan_requeue_unit.py`
- `recommendation`: Route all workflow continuation resumes through a durable Jobs or Scheduler task keyed by run ID, step ID, and resume cause; record the continuation intent before acknowledging approval/checkpoint/retry success; mark research wait links resumed only after the continuation is accepted by durable ownership or reaches a terminal handoff state; add regression tests for continuation task failure, process shutdown before task execution, and duplicate resume dedupe.
- `status`: `open`
- `validation_status`: `needs_reproduction`

Existing normalized findings confirmed or escalated by this pass:

- Confirmed: `AUDIT-2026-06-27-AUTH-003`, `AUDIT-2026-06-27-DB-001`, `AUDIT-2026-06-27-DB-002`, `AUDIT-2026-06-27-WEBUI-002`, `AUDIT-2026-06-27-OPS-001`, `AUDIT-2026-06-27-OPS-002`, `AUDIT-2026-06-27-OPS-006`, `AUDIT-2026-06-27-MEDIA-003`, `AUDIT-2026-06-27-JOBS-001`, `AUDIT-2026-06-27-JOBS-002`, `AUDIT-2026-06-27-MCP-002`.
- Recommended reliability follow-up: `AUDIT-2026-06-27-INTEGRATIONS-001`, `AUDIT-2026-06-27-INTEGRATIONS-002`, and `AUDIT-2026-06-27-INTEGRATIONS-003` should be verified with outbound-policy/proxy tests because raw HTTP clients also bypass centralized retry, timeout, proxy, and logging controls.
- No existing normalized finding was refuted.

## Confirmed Issues

No new specialist-specific confirmed issues are added beyond the normalized index.

Confirmed existing issues:

- `AUDIT-2026-06-27-DB-001`: Confirmed from the DB domain report, reproduction evidence, and `sqlite_helpers.py` routing older file-backed SQLite Media DBs through the generic package migration directory. The package directory contains Prompt Studio and v23 Media DB scripts but lacks a contiguous Media DB chain for v9 through v22, so older Media DB startup/upgrade remains data-durability blocking.
- `AUDIT-2026-06-27-DB-002`: Confirmed from reproduction evidence and `db_migration.py`. Migration SQL, success ledger writes, and `schema_version` updates are separate commit windows, and the reproduced failing multi-statement script left the first DDL table behind while recording the migration as failed.
- `AUDIT-2026-06-27-WEBUI-002`: Confirmed as a streaming lifecycle/API-contract issue. The frontend opens TTS WebSocket streams with query-token auth, while the backend rejects query tokens by default and expects header or initial auth-frame authentication.
- `AUDIT-2026-06-27-OPS-001`: Confirmed as a release reliability gate gap. Worker and audio-worker images are published but not built by the PR container gate.
- `AUDIT-2026-06-27-OPS-006`: Confirmed as an operations reliability issue. The Kubernetes sample uses a literal `${POSTGRES_PASSWORD}` inside `DATABASE_URL`, so applying it as-is creates a database credential mismatch.
- `AUDIT-2026-06-27-JOBS-001`: Confirmed and escalated by the workflow continuation finding below. The current workflow run path persists rows, then hands actual execution to in-process queues and daemon threads with no durable owner or startup repair for queued runs.
- `AUDIT-2026-06-27-MCP-002`: Confirmed from `agent_client_protocol.py`, `ws_broadcaster.py`, and `event_bus.py`. The reconnect branch starts a local `WSBroadcaster` and connection but the endpoint finalizer only unregisters the runner callback and stops the stream, leaving the broadcaster task/subscription path unmanaged.

## Likely Risks

### AUDIT-2026-06-27-REL-001 - Workflow continuation resumes are fire-and-forget tasks outside durable scheduler ownership

- `severity`: `medium`
- `confidence`: `high`
- `category`: `reliability`
- `evidence_tier`: `likely_risk`
- `evidence_strength`: `static_confirmed`
- `status`: `open`
- `validation_status`: `needs_reproduction`
- `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/reliability-lifecycle.md`
- `owner_domain`: `Reliability and async lifecycle`
- `affected_paths`:
  - `tldw_Server_API/app/api/v1/endpoints/workflows.py`
  - `tldw_Server_API/app/api/v1/endpoints/research_runs.py`
  - `tldw_Server_API/app/core/Workflows/engine.py`
  - `tldw_Server_API/app/core/Workflows/research_wait_bridge.py`
  - `tldw_Server_API/tests/Workflows/test_workflows_api.py`
  - `tldw_Server_API/tests/Workflows/test_versions_idempotency.py`
  - `tldw_Server_API/tests/Workflows/test_orphan_requeue_unit.py`
- `evidence`:
  - Workflow run retry and generic run control spawn `engine.continue_run(...)` through bare `asyncio.create_task(...)` in `workflows.py` rather than submitting a durable Scheduler/Jobs task.
  - Approval and rejection continuations also create local tasks after mutating approval/rejection state.
  - The orphan reaper marks stale step rows failed, updates the run to `running` with `status_reason="orphan_requeued"`, appends `run_requeued`, and then creates a local continuation task.
  - `research_wait_bridge._schedule_resume()` returns `asyncio.create_task(engine.continue_run(...))`; `resume_workflows_waiting_on_research_checkpoint()` marks the research wait link resumed immediately after scheduling, not after the continuation completes or is durably accepted.
  - `research_runs.patch_and_approve_research_checkpoint()` suppresses bridge scheduling exceptions and only yields once after creating the background task.
  - Existing tests cover successful retry/approval continuation and the case where `_schedule_resume` raises before scheduling. I did not find coverage for a continuation task that is accepted by `create_task` and then fails, is cancelled on shutdown, or is lost before execution.
- `impact`: A process crash, request-loop teardown, cancellation, or exception after approval/retry/checkpoint acknowledgement can leave a workflow run or research wait in a state that says it was resumed/requeued while the actual continuation did not run to completion. Unlike `core_scheduler.submit(...)`, these tasks have no durable task ID, no idempotency key, no worker ownership, and no retry/repair loop. This is related to `AUDIT-2026-06-27-JOBS-001` but materially different because these continuation paths bypass even the in-process `WorkflowScheduler.submit(...)` handoff.
- `recommendation`: Persist a continuation intent and submit it through Jobs or Scheduler with a deterministic idempotency key based on run ID, step ID, continuation cause, and target step. Treat approvals, rejections, retries, orphan requeues, and research checkpoint resumes as accepted only after the durable continuation owner records the handoff. Add tests for post-schedule task failure, shutdown before task execution, duplicate resume, and research wait retryability when continuation execution fails after task creation.

Confirmed existing likely risks:

- `AUDIT-2026-06-27-AUTH-003`: Confirmed as a PostgreSQL reliability risk. The endpoint uses raw `pool.acquire()` plus `?` placeholders, while the placeholder conversion layer is in `DatabasePool.execute()` and `DatabasePool.fetchone()`.
- `AUDIT-2026-06-27-MEDIA-003`: Confirmed as a data-durability risk. Original file storage writes the permanent file before `db.insert_media_file(...)`; the catch block marks storage failed but does not call the filesystem backend's `delete()`.
- `AUDIT-2026-06-27-JOBS-002`: Confirmed as a reliability risk. Other recurring schedulers sampled use deterministic idempotency keys, while workflow and ACP schedule fires submit core Scheduler tasks without `idempotency_key`.
- `AUDIT-2026-06-27-OPS-002`: Confirmed as an operational hardening risk for worker images. It is security-categorized in the index, but the reliability angle is that runtime image drift and root execution are not smoke-checked for the published worker artifacts.

## Improvement Opportunities

No new specialist-specific improvement-opportunity findings are added.

Recommended follow-up from existing findings:

- `AUDIT-2026-06-27-INTEGRATIONS-001`, `AUDIT-2026-06-27-INTEGRATIONS-002`, and `AUDIT-2026-06-27-INTEGRATIONS-003`: Add centralized HTTP client tests that assert timeout/retry/proxy/egress behavior, not only security blocking. These are security-categorized in the index, but the same bypasses also reduce reliability consistency.
- For `AUDIT-2026-06-27-JOBS-001` and `AUDIT-2026-06-27-REL-001`, prefer one workflow execution ownership model. Mixing daemon threads, request-loop tasks, APScheduler, core Scheduler tasks, and Jobs makes shutdown and replay semantics difficult to reason about.
- For `AUDIT-2026-06-27-MCP-002`, add endpoint-level reconnect-disconnect tests that assert no event-bus subscriber and no broadcaster task remain after disconnect.

## Coverage And Evidence

### Files Inspected

- `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md`
- All domain reports under `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/`
- Existing evidence files: `db-migrations-data-durability-reproductions.txt`, `ci-deployment-operations-release-candidates.txt`, `integrations-providers-static-evidence.txt`, `webui-extension-api-contracts-static-evidence.txt`, `backend-test-inventory.txt`, `db-migration-inventory.txt`, `bandit-app-summary.txt`
- Lifecycle and worker orchestration: `tldw_Server_API/app/main.py`, `lifespan_startup_sequence.py`, `lifespan_shutdown_sequence.py`, `shutdown_coordinator.py`, `lifecycle_worker_engine.py`, `lifecycle_workers.py`, `startup_worker_bootstrap.py`, `startup_worker_groups.py`, `startup_recurring_schedulers.py`, `lifecycle_worker_startup_adapters.py`, `shutdown_owned_job_pollers.py`, `startup_cleanup_workers.py`
- Workflows, Jobs, and Scheduler: `workflows.py`, `research_runs.py`, `workflows_scheduler.py`, `workflows_db_maintenance.py`, `core/Workflows/engine.py`, `core/Workflows/research_wait_bridge.py`, `core/Scheduler/scheduler.py`, `core/Scheduler/backends/sqlite_backend.py`, `core/Scheduler/backends/postgresql_backend.py`, `core/Scheduler/handlers/workflows.py`, selected `core/Jobs` and worker SDK paths via domain reports
- Resource cleanup and durability paths: `agent_client_protocol.py`, `ws_broadcaster.py`, `event_bus.py`, `db_migration.py`, `sqlite_helpers.py`, `persistence.py`, `filesystem_storage.py`, `admin_impersonation.py`, `AuthNZ/database.py`
- Tests sampled or inventoried: `tests/Services/test_lifecycle_worker_engine.py`, `test_lifespan_shutdown_sequence.py`, `test_shutdown_owned_job_pollers.py`, `test_startup_recurring_schedulers.py`, `tests/Workflows/test_workflows_scheduler.py`, `test_orphan_requeue_unit.py`, `test_workflows_api.py`, `test_versions_idempotency.py`, `tests/Agent_Client_Protocol/test_ws_reconnect.py`, `tests/DB_Management/test_db_migration_*`, `tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py`, and broad lifecycle/job/scheduler test inventory searches.

### Tests Or Scans Run

- Static inspection commands using `sed`, `find`, `wc`, `jq`, and `rg` over the normalized index, every domain report, lifecycle source, workflow source, DB/media durability source, and relevant test inventories.
- No new runtime pytest suite was run for this specialist pass. The pass consumed prior domain test evidence, including DB migration reproductions, focused Jobs/Workflow tests, MCP WebSocket tests, media storage tests, and integration/provider tests recorded in the domain reports.
- No Bandit scan was rerun because this pass changed only the specialist report and did not change production code.

### Blocked Or Unverified Areas

- Multi-process APScheduler duplication, process-kill recovery, request-loop shutdown races, and post-`create_task` continuation failure were not runtime-reproduced in this specialist pass.
- PostgreSQL-specific behavior for `AUDIT-2026-06-27-AUTH-003` was statically confirmed but not reproduced against a live PostgreSQL fixture.
- Container image build/runtime checks for `AUDIT-2026-06-27-OPS-001` and `AUDIT-2026-06-27-OPS-002` were not run because Docker/service execution was outside this assignment.
- Full repository tests were not run.
- No network access, dependency installation, Docker, service startup, production code edits, Backlog task edits, staging, or commits were performed.

### Evidence Notes

- The main application lifecycle has substantial positive controls: startup worker specs are validated for duplicate names, dependencies, phases, and startup rollback; shutdown runs phased worker stop with timeouts; recurring schedulers are registered with lifecycle callbacks; job pollers set stop events and are quiesced after the Jobs acquire gate is enabled.
- Most recurring schedulers sampled use deterministic idempotency keys (`reading_digest`, `reminders`, `companion_reflection`, `admin_backup`, `ingestion_sources`). This supports keeping `AUDIT-2026-06-27-JOBS-002` focused on workflow/ACP recurring schedule fires rather than broadening it to all schedulers.
- The new `AUDIT-2026-06-27-REL-001` finding should be reconciled with `AUDIT-2026-06-27-JOBS-001` during index finalization. If the coordinator chooses not to add a new index entry, `AUDIT-2026-06-27-JOBS-001` should be expanded to explicitly include `continue_run` fire-and-forget continuation paths and research checkpoint resume marking.
