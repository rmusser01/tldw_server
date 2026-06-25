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
updated_date: 2026-06-25 03:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the hardened Claims Jobs Stage 1 plan with subagent-driven TDD: contracts, enqueue helpers, rebuild/notification/alert job handlers, service routing, worker lifecycle registration, integration verification, and security sweep. Keep Jobs as the only queue/lifecycle owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Task 1 Claims job contracts and payload validation implemented with tests.
- [ ] #2 Task 2 enqueue helpers and read-only Jobs summary implemented with tests.
- [ ] #3 Task 3 rebuild business seam implemented with tests.
- [ ] #4 Task 4 monitoring event reload support implemented with tests.
- [ ] #5 Task 5 review notification delivery seam implemented with tests.
- [ ] #6 Task 6 Claims job handlers implemented with tests.
- [ ] #7 Task 7 Claims service routing to Jobs implemented with tests.
- [ ] #8 Task 8 Claims Jobs worker lifecycle registration implemented with tests.
- [ ] #9 Task 9 integration verification and Bandit security sweep complete.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 complete. Implemented Claims Jobs contract constants, payload validation, WorkerSDK-compatible ClaimsJobError, and result helpers with strict ID-only top-level allowlists and owner validation precedence. Review loop fixed bool/float numeric coercion, non-scalar owners, unknown-key handling, invalid JSON/non-object JSON coverage, and reserved result-field overrides. Verification: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py -q => 29 passed, 70 warnings. Spec review: compliant. Code-quality review: ready to proceed. Commits: fb8b7f772e, 45b436e2c0, 76331f44f2, 2c9edcb42a.
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
