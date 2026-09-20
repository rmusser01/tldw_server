---
id: TASK-12988
title: Reconcile Jobs admission counter baseline and close lifecycle stream
status: In Progress
created_date: 2026-07-27 15:16
labels:
- Jobs
- tests
- stability
- postgres
- sqlite
priority: medium
references:
- TASK-12969.3
- https://github.com/rmusser01/tldw_server/pull/2763
- origin/dev 616d6dd35d48849f22b320d34823bfcfecbc4b74
- https://github.com/rmusser01/tldw_server/pull/2765
- branch commit f9a5b8e733c18be1e4c73a374ac82d932a68284e
documentation:
- Docs/superpowers/plans/2026-07-14-jobs-admission-hardening-and-lease-lifecycle.md
- Docs/superpowers/specs/2026-06-24-jobs-backend-parity-refactor-design.md
- Docs/superpowers/plans/2026-07-27-jobs-admission-counter-baseline-reconciliation.md
modified_files:
- tldw_Server_API/tests/Jobs/test_jobs_dependency_acquire_counter_regressions.py
- Docs/superpowers/plans/2026-07-14-jobs-admission-hardening-and-lease-lifecycle.md
- Docs/superpowers/plans/2026-07-27-jobs-admission-counter-baseline-reconciliation.md
- backlog/tasks/task-12969.3 - Extract-Jobs-single-job-lease-renewal-and-release-operations.md
updated_date: 2026-07-27 15:34
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Align the admission counter failure regression tests with the already-merged backend-specific durability contracts. This is a test-and-tracking stabilization slice only: PostgreSQL treats optional counter updates as best-effort behind a savepoint, while SQLite keeps counter updates transaction-critical. Close the merged lease renewal/release child task and plan stage with evidence, without changing production behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SQLite admission counter failure still raises and rolls back the job and durable created event.
- [x] #2 PostgreSQL admission counter failure is isolated by a savepoint; the job and durable created event commit while the failed counter is not advanced.
- [x] #3 Focused admission and neighboring Jobs matrices pass with required PostgreSQL enabled and no skips or known deselections.
- [x] #4 TASK-12969.3 and the 2026-07-14 implementation plan are finalized with PR #2763 merge evidence.
- [x] #5 The duplicate TASK-12969 parent-ID collision is documented and the ambiguous parent records remain unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the stale PostgreSQL expectations on clean origin/dev and inspect existing helper assertions.
2. Split the backend-specific regression expectations without changing production code.
3. Run focused and neighboring Jobs verification, Ruff, Bandit, and diff checks.
4. Finalize TASK-12969.3 and the lifecycle plan with merge evidence; document the TASK-12969 ID collision.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
No production files are in scope. The project currently has four records with ID TASK-12969, so the parent cannot be safely mutated through the Backlog MCP; record the collision rather than editing any ambiguous parent.
Baseline on clean origin/dev 616d6dd35d reproduced two SQLite passes and two required PostgreSQL failures (`Failed: DID NOT RAISE RuntimeError`). The test was split into explicit backend contracts with no production change. Focused contract: 4 passed, zero skips. Admission-focused matrix: 110 passed, zero skips. Full neighboring matrix: 104 passed, zero skips or deselections. Ruff, compileall, and git diff --check pass. Full Bandit reports 110 expected test-only B101 assertion findings and zero errors; a second scan excluding B101 reports zero findings and zero errors. TASK-12969.3 is Done and the prior plan's Stage 5/Task 13 Step 4 are complete with PR #2763 merge evidence. Four files still claim TASK-12969, so all ambiguous parent records remain untouched.
Commit f9a5b8e733 was pushed on codex/jobs-admission-baseline-reconcile and draft PR #2765 opened against dev. The task remains In Progress until merge. The PR is merge-blocked by the requester-owned Change summary placeholder required by repository policy.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reconciled the Jobs admission counter regression baseline without changing production behavior. SQLite now explicitly proves that a counter failure rolls back the admitted job and durable created event; PostgreSQL explicitly proves that its savepoint-isolated optional counter failure still commits the job and created event while leaving the counter unpersisted. Finalized the already-merged TASK-12969.3 lifecycle extraction and plan stage with PR #2763 merge evidence, and documented rather than mutating the four ambiguous TASK-12969 parent records. Draft PR #2765 contains the stabilization change and remains pending requester summary and merge.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
