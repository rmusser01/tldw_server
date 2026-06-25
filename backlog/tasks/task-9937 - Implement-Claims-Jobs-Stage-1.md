---
id: TASK-9937
title: Implement Claims Jobs Stage 1
status: In Progress
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
updated_date: 2026-06-25 05:10
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
- [ ] #6 Task 6 Claims job handlers implemented with tests.
- [ ] #7 Task 7 Claims service routing to Jobs implemented with tests.
- [ ] #8 Task 8 Claims Jobs worker lifecycle registration implemented with tests.
- [ ] #9 Task 9 integration verification and Bandit security sweep complete.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 complete. Implemented Claims Jobs contract constants, payload validation, WorkerSDK-compatible ClaimsJobError, and result helpers with strict ID-only top-level allowlists and owner validation precedence. Review loop fixed bool/float numeric coercion, non-scalar owners, unknown-key handling, invalid JSON/non-object JSON coverage, and reserved result-field overrides. Verification: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py -q => 29 passed, 70 warnings. Spec review: compliant. Code-quality review: ready to proceed. Commits: fb8b7f772e, 45b436e2c0, 76331f44f2, 2c9edcb42a.
Task 2 complete. Implemented Claims Jobs enqueue helpers, read-only summary helper, feature/worker config helpers, and strict Jobs delegation for rebuild, review notification, and alert jobs. Review loop refined rebuild enqueue to be repeatable by default with optional bounded idempotency_scope, added environment fallback for CLAIMS_JOBS_* config while preserving explicit settings_obj precedence, clamped negative retry values to defaults, and expanded alert success coverage. Verification: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py -q => 38 passed, 88 warnings. Spec review: compliant. Code-quality review: ready to proceed. Commits: 8743d44dc0, db7a2104f6.
Task 3 complete. Extracted the one-media rebuild body into rebuild_claims_for_media with small non-sensitive result dictionaries for ok and skipped outcomes, kept _process_task as a legacy delegate, and preserved rollback behavior when replacement claim storage inserts zero rows. Review loop added direct coverage for the successful helper result contract. Verification: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Claims/test_claims_rebuild_service_failure.py -q => 7 passed, 26 warnings. Spec review: compliant. Code-quality review: ready to proceed after minor test gap resolved. Commits: 25dca6d9dd, f8ca1f720f.
Task 4 complete. Updated Claims monitoring event DB helpers so insert_claims_monitoring_event returns the inserted row, added get_claims_monitoring_event lookup and MediaDatabase binding, and covered both SQLite and fake PostgreSQL RETURNING-id paths. Verification: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_claims_monitoring_event_ops.py -q => 7 passed, 26 warnings. Spec review: compliant. Code-quality review: ready to proceed after PostgreSQL branch coverage was added. Commits: 4a5594af20, 99ea3aec0c.
Task 5 complete. Extracted deliver_claim_review_notifications_now as a synchronous review-notification delivery seam with small result dictionaries, kept legacy dispatch on bounded submission, and preserved legacy DB initialization by having dispatch pass initialize=True while the Jobs-facing helper defaults to initialize=False. Review loop fixed mixed delivered/pending batches to mark only pending IDs and hardened ID normalization for bool, invalid, and overflow inputs. Verification: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Claims/test_claims_review_notifications.py -q => 10 passed, 32 warnings. Spec review: compliant under accepted legacy-dispatch initialization refinement. Code-quality review: ready to proceed. Commits: ae7c6047c2, 3a7b559cdd, 2eeb96a272.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Backlog task kept current with implementation notes and touched files.
- [ ] #2 Each plan task follows TDD red/green/refactor and commits its bounded slice.
- [ ] #3 Spec and code-quality review completed after each implementation task.
- [ ] #4 Focused pytest commands pass for touched Claims/Jobs scope.
- [ ] #5 Bandit run on touched code scope and new findings addressed.
- [ ] #6 Final git diff reviewed and implementation summary recorded.
<!-- DOD:END -->
