---
id: TASK-9937
title: Implement Claims Jobs Stage 1
status: Done
created_date: 2026-06-25 03:18
labels:
- claims
- jobs
- refactor
- implementation
priority: high
references:
- Docs/superpowers/specs/2026-06-24-claims-jobs-operational-control-plane-design.md
- Docs/superpowers/plans/2026-06-25-claims-jobs-stage1-implementation-plan.md
- TASK-9935
- TASK-9936
documentation:
- Docs/superpowers/plans/2026-06-25-claims-jobs-stage1-implementation-plan.md
modified_files:
- backlog/tasks/task-9937 - Implement-Claims-Jobs-Stage-1.md
- tldw_Server_API/app/core/Claims_Extraction/claims_job_contracts.py
- tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py
- tldw_Server_API/app/core/Claims_Extraction/claims_jobs.py
- tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py
- tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py
- tldw_Server_API/tests/Claims/test_claims_rebuild_service_failure.py
- tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_monitoring_event_ops.py
- tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py
- tldw_Server_API/tests/DB_Management/test_media_db_claims_monitoring_event_ops.py
- tldw_Server_API/app/core/Claims_Extraction/claims_notifications.py
- tldw_Server_API/tests/Claims/test_claims_review_notifications.py
- tldw_Server_API/app/core/Claims_Extraction/claims_job_handlers.py
- tldw_Server_API/app/core/Claims_Extraction/claims_alert_delivery.py
- tldw_Server_API/app/core/Claims_Extraction/claims_service.py
- tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py
- tldw_Server_API/tests/Claims/test_claims_webhook_delivery.py
- tldw_Server_API/app/core/Claims_Extraction/ingestion_claims.py
- tldw_Server_API/app/api/v1/schemas/claims_schemas.py
- tldw_Server_API/tests/Claims/test_claims_review_api.py
- tldw_Server_API/tests/Claims/test_claims_rebuild_stale_policy.py
- tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py
- tldw_Server_API/tests/Claims/test_ingestion_claims_sql.py
- tldw_Server_API/app/services/claims_jobs_worker.py
- tldw_Server_API/app/services/startup_worker_groups.py
- tldw_Server_API/tests/Services/test_claims_jobs_worker.py
- tldw_Server_API/tests/Services/test_startup_worker_groups.py
updated_date: 2026-06-25 07:27
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the hardened Claims Jobs Stage 1 plan with subagent-driven TDD: contracts, enqueue helpers, rebuild/notification/alert job handlers, service routing, worker lifecycle registration, integration verification, and security sweep. Keep Jobs as the only queue/lifecycle owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Task 1 Claims job contracts and payload validation implemented with tests.
- [x] #2 Task 2 enqueue helpers and read-only Jobs summary implemented with tests.
- [x] #3 Task 3 rebuild business seam implemented with tests.
- [x] #4 Task 4 monitoring event reload support implemented with tests.
- [x] #5 Task 5 review notification delivery seam implemented with tests.
- [x] #6 Task 6 Claims job handlers implemented with tests.
- [x] #7 Task 7 Claims service routing to Jobs implemented with tests.
- [x] #8 Task 8 Claims Jobs worker lifecycle registration implemented with tests.
- [x] #9 Task 9 integration verification and Bandit security sweep complete.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 complete. Implemented Claims Jobs contract constants, payload validation, WorkerSDK-compatible ClaimsJobError, and result helpers with strict ID-only top-level allowlists and owner validation precedence. Review loop fixed bool/float numeric coercion, non-scalar owners, unknown-key handling, invalid JSON/non-object JSON coverage, and reserved result-field overrides. Verification: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py -q => 29 passed, 70 warnings. Spec review: compliant. Code-quality review: ready to proceed. Commits: fb8b7f772e, 45b436e2c0, 76331f44f2, 2c9edcb42a.
Task 2 complete. Implemented Claims Jobs enqueue helpers, read-only summary helper, feature/worker config helpers, and strict Jobs delegation for rebuild, review notification, and alert jobs. Review loop refined rebuild enqueue to be repeatable by default with optional bounded idempotency_scope, added environment fallback for CLAIMS_JOBS_* config while preserving explicit settings_obj precedence, clamped negative retry values to defaults, and expanded alert success coverage. Verification: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py -q => 38 passed, 88 warnings. Spec review: compliant. Code-quality review: ready to proceed. Commits: 8743d44dc0, db7a2104f6.
Task 3 complete. Extracted the one-media rebuild body into rebuild_claims_for_media with small non-sensitive result dictionaries for ok and skipped outcomes, kept _process_task as a legacy delegate, and preserved rollback behavior when replacement claim storage inserts zero rows. Review loop added direct coverage for the successful helper result contract. Verification: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Claims/test_claims_rebuild_service_failure.py -q => 7 passed, 26 warnings. Spec review: compliant. Code-quality review: ready to proceed after minor test gap resolved. Commits: 25dca6d9dd, f8ca1f720f.
Task 4 complete. Updated Claims monitoring event DB helpers so insert_claims_monitoring_event returns the inserted row, added get_claims_monitoring_event lookup and MediaDatabase binding, and covered both SQLite and fake PostgreSQL RETURNING-id paths. Verification: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_claims_monitoring_event_ops.py -q => 7 passed, 26 warnings. Spec review: compliant. Code-quality review: ready to proceed after PostgreSQL branch coverage was added. Commits: 4a5594af20, 99ea3aec0c.
Task 5 complete. Extracted deliver_claim_review_notifications_now as a synchronous review-notification delivery seam with small result dictionaries, kept legacy dispatch on bounded submission, and preserved legacy DB initialization by having dispatch pass initialize=True while the Jobs-facing helper defaults to initialize=False. Review loop fixed mixed delivered/pending batches to mark only pending IDs and hardened ID normalization for bool, invalid, and overflow inputs. Verification: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Claims/test_claims_review_notifications.py -q => 10 passed, 32 warnings. Spec review: compliant under accepted legacy-dispatch initialization refinement. Code-quality review: ready to proceed. Commits: ae7c6047c2, 3a7b559cdd, 2eeb96a272.
Task 6 complete. Added Claims job handlers for rebuild, review notification, and alert-delivery jobs using the Claims domain seams while keeping queue/lifecycle mechanics in Jobs. Review loop hardened owner scoping by canonicalizing owner IDs before comparison/DB path derivation, filtering review notification rows by owner before delivery/marking, and adding regression coverage for alert event owner mismatch, alert owner mismatch, already-delivered skip, and retryable webhook failure. Verification: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py tldw_Server_API/tests/Claims/test_claims_webhook_delivery.py tldw_Server_API/tests/Claims/test_claims_review_notifications.py -q => 25 passed, 62 warnings; /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Claims/test_claims_rebuild_service_failure.py -q => 7 passed, 26 warnings; /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m ruff check touched Task 6 files => all checks passed; Bandit touched Task 6 production scope => 0 findings. Spec re-review: compliant. Code-quality re-review: PASS. Commits: 1ec28befe3, e2e39f1e39.
Task 7 complete. Routed Claims service-facing background work through the Jobs module without adding Claims-side queue controls: explicit rebuild and rebuild-all enqueue Claims rebuild Jobs when enabled, review and assignment notifications enqueue review notification Jobs best-effort, alert delivery enqueues Jobs for valid persisted event IDs, and dashboard analytics exposes a read-only Claims Jobs summary. Review loop fixed Jobs enqueue failure handling for PostgreSQL/storage exceptions, ownerless rebuild fallback behavior, alert delivery malformed event-row handling, and rebuild-all idempotency by using a bounded scope that dedupes immediate retries while allowing future operations. Verification: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Claims/test_ingestion_claims_sql.py tldw_Server_API/tests/Claims/test_claims_review_api.py tldw_Server_API/tests/Claims/test_claims_rebuild_stale_policy.py tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py -q => 22 passed, 209 warnings; Ruff touched Task 7 files => all checks passed; Bandit touched Task 7 production scope => 0 findings. Spec re-review: PASS. Code-quality re-review: PASS. Commits: 2a9030339d, da4f03421b, f87000d57e, 3fb9238a7a, 2d6718ae5b, 0dec33cc8c, d424091f0b.
Task 8 complete. Added a lifecycle-managed Claims Jobs worker that builds a WorkerSDK config from existing Claims Jobs helpers, runs process_claims_job without an owner_user_id filter so Claims jobs are acquired across all owners in the Claims domain/queue, and stops through the lifecycle stop event. Registered the Claims Jobs provider before the legacy Claims rebuild provider and updated service tests to document provider order and the real spec graph. Red verification: the new worker test initially failed with ImportError for missing claims_jobs_worker, and the startup graph test failed because claims_jobs_task was absent. Green verification: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Services/test_claims_jobs_worker.py tldw_Server_API/tests/Services/test_startup_worker_groups.py -q => 5 passed, 22 warnings. Ruff touched Task 8 files => all checks passed. Bandit touched Task 8 production files => 0 findings in /tmp/bandit_claims_task8.json.
Task 9 complete. Final Claims Jobs Stage 1 verification passed after lint cleanup. Focused Claims Jobs suite: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py tldw_Server_API/tests/Claims/test_claims_review_notifications.py tldw_Server_API/tests/Claims/test_claims_webhook_delivery.py tldw_Server_API/tests/Services/test_claims_jobs_worker.py -q => 65 passed, 142 warnings. Related regression suite: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Claims/test_claims_review_api.py tldw_Server_API/tests/Claims/test_claims_rebuild_stale_policy.py tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py tldw_Server_API/tests/Claims/test_claims_alerts_scheduler.py tldw_Server_API/tests/DB_Management/test_media_db_claims_monitoring_event_ops.py tldw_Server_API/tests/Services/test_startup_worker_groups.py -q => 28 passed, 173 warnings. Jobs owner/idempotency guard: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_rls_postgres.py -q => 2 passed, 2 skipped, 16 warnings; exact skip reason: Jobs tests run only in the jobs-suite CI workflow. Final Ruff touched Stage 1 files => all checks passed. Final Bandit: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Claims_Extraction tldw_Server_API/app/services/claims_jobs_worker.py -f json -o /tmp/bandit_claims_jobs_stage1.json => 0 findings, 0 errors. git diff --check => no whitespace errors. Final cleanup also made ruff-sorted imports in touched DB helper files and narrowed one test from pytest.raises(Exception) to ClaimsJobError.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Claims Jobs Stage 1 is complete. The Claims module now defines ID-only Jobs contracts, enqueue helpers, domain handlers, service-routing integration, a WorkerSDK-based lifecycle worker, and read-only dashboard Jobs status while keeping queue lifecycle, leases, retries, and admin controls inside the core Jobs module. Verified with focused Claims Jobs suites, related Claims regressions, Jobs idempotency/owner guard tests where available, final Ruff, Bandit, and whitespace checks. PostgreSQL RLS Jobs tests were skipped locally because they are restricted to the jobs-suite CI workflow.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Backlog task kept current with implementation notes and touched files.
- [x] #2 Each plan task follows TDD red/green/refactor and commits its bounded slice.
- [x] #3 Spec and code-quality review completed after each implementation task.
- [x] #4 Focused pytest commands pass for touched Claims/Jobs scope.
- [x] #5 Bandit run on touched code scope and new findings addressed.
- [x] #6 Final git diff reviewed and implementation summary recorded.
<!-- DOD:END -->
