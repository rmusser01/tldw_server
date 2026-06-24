# Jobs Backend Parity Inventory

Date: 2026-06-24
Source spec: Docs/superpowers/specs/2026-06-24-jobs-backend-parity-refactor-design.md

## Purpose

This inventory defines the compatibility boundary for the first Jobs backend parity refactor PR. Production SQL extraction must not begin until each state-changing or public-facing path below is either covered by tests in this PR or explicitly assigned to a later extraction slice.

## Direct Runtime Jobs SQL

| Area | File | Representative SQL | Classification | First Slice Action |
| --- | --- | --- | --- | --- |
| Jobs admin SLA policies | `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py` | `SELECT * FROM job_sla_policies` | read-only/status SQL | Defer as read model; existing SLA endpoint tests remain coverage. |
| Jobs admin SLA breaches | `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py` | `SELECT id, domain, queue, job_type, status FROM jobs` | read-only/status SQL | Defer as read model; no extraction in first slice. |
| Jobs admin archive metadata | `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py` | `SELECT payload, result, payload_compressed, result_compressed FROM jobs_archive` | read-only/status SQL | Defer as read model; no extraction in first slice. |
| Jobs admin stale processing (GET /jobs/stale, jobs_admin.py:1513) | `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py` | `SELECT domain, queue, COUNT(*) FROM jobs` | read-only/status SQL | Defer as read model; cover in stale/admin status contract slice. |
| Jobs admin batch cancel | `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py` | `UPDATE jobs SET status='cancelled'` | state-changing SQL | Defer extraction; keep existing endpoint behavior and require separate operation slice. |
| Jobs admin batch reschedule | `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py` | `UPDATE jobs SET available_at = NOW() + interval` | state-changing SQL | Defer extraction; keep existing endpoint behavior and require separate operation slice. |
| Jobs admin requeue quarantined | `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py` | `UPDATE jobs SET status='queued'` | state-changing SQL | Defer extraction; keep existing endpoint behavior and require separate operation slice. |
| Prompt Studio status dashboard | `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_status.py` | `SELECT status, COUNT(*) AS c FROM jobs` | read-only/status SQL | Defer as read model; cover in status dashboard contract slice. |
| Jobs metrics service | `tldw_Server_API/app/services/jobs_metrics_service.py` | `SELECT COUNT(*) FROM jobs` | service/worker operational SQL | Defer; keep service-specific metrics tests. |
| Audio jobs worker fairness scans | `tldw_Server_API/app/services/audio_jobs_worker.py` | `SELECT owner_user_id FROM jobs` | service/worker operational SQL | Defer; cover in worker-specific slice if acquire semantics move. |
| Jobs webhooks service | `tldw_Server_API/app/services/jobs_webhooks_service.py` | `SELECT id, event_type FROM job_events` | service/worker operational SQL | Defer; event outbox extraction owns this boundary. |
| External sources quota scan | `tldw_Server_API/app/core/External_Sources/connectors_service.py` | `SELECT COUNT(*) AS c FROM jobs` | service/worker operational SQL | Defer; not part of admission/lifecycle first slice. |

## Domain Status And Identifier Mappings

| Domain | Endpoint Or Adapter | Mapping | First Slice Action |
| --- | --- | --- | --- |
| Embeddings | `tldw_Server_API/app/core/Embeddings/jobs_adapter.py` | `quarantined -> failed`; unknown status derives as `processing`; public id prefers `jobs.uuid` | Defer endpoint contract; existing adapter tests stay active. |
| Chatbooks export | `tldw_Server_API/app/core/Chatbooks/jobs_adapter.py` | `queued -> pending`; `processing -> in_progress`; `quarantined -> failed`; payload `chatbooks_job_id` preferred over Jobs id | Add adapter contract tests in this PR. |
| Chatbooks import | `tldw_Server_API/app/core/Chatbooks/jobs_adapter.py` | `queued -> pending`; `processing -> in_progress`; `quarantined -> failed`; payload `chatbooks_job_id` preferred over Jobs id | Add adapter contract tests in this PR. |
| Prompt Studio optimization | `tldw_Server_API/app/core/Prompt_Management/prompt_studio/jobs_adapter.py` | `quarantined -> failed`; unknown status falls back to `queued` | Defer to domain adapter slice; no production extraction in first PR. |

## First PR Compatibility Gates

- Shared SQLite/Postgres scenarios cover idempotent create, acquire, renew stale/no-op behavior, complete idempotency, cancel terminal no-op, and events outbox behavior.
- Admin list/detail public responses are tested by required fields, not snapshots.
- Chatbooks adapter mapping is tested without FastAPI startup.
- `JobsSettings` documents snapshot, refresh, and operation-time setting groups before manager integration.
- Operation contract dataclasses exist and do not import `JobManager`.
