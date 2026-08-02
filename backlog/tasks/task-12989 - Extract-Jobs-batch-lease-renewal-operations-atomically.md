---
id: TASK-12989
title: Extract Jobs batch lease renewal operations atomically
status: In Progress
created_date: 2026-08-02 02:14
labels:
- jobs
- refactor
- stability
- postgresql
- sqlite
priority: medium
references:
- TASK-12988
- Docs/superpowers/specs/2026-06-24-jobs-backend-parity-refactor-design.md
- Docs/superpowers/plans/2026-07-14-jobs-admission-hardening-and-lease-lifecycle.md
- https://github.com/rmusser01/tldw_server/pull/2765
documentation:
- Docs/superpowers/specs/2026-08-01-jobs-batch-lease-renewal-extraction-design.md
modified_files:
- tldw_Server_API/app/core/Jobs/manager.py
- tldw_Server_API/app/core/Jobs/operations/contracts.py
- tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py
- tldw_Server_API/app/core/Jobs/operations/postgres/lifecycle.py
- tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py
- tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py
- tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py
updated_date: 2026-08-02 02:17
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Characterize and extract JobManager.batch_renew_leases into dedicated SQLite and PostgreSQL lifecycle operations while preserving the public facade, input-order processing, exact row-count semantics, backend-specific clock timing, expected no-op behavior, and one atomic transaction for the complete batch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 JobManager.batch_renew_leases keeps its public signature and integer return contract.
- [ ] #2 SQLite and PostgreSQL batch renewal route through dedicated backend lifecycle operations that do not import JobManager.
- [ ] #3 Unexpected backend or clock failures roll back every earlier renewal in the batch.
- [ ] #4 Missing, non-processing, and stale-lease items remain non-fatal no-ops while valid items commit.
- [ ] #5 Duration clamping, duplicate-item counting, empty-batch behavior, and backend-specific clock timing remain compatible.
- [ ] #6 Focused SQLite and required real PostgreSQL suites pass with zero PostgreSQL skips.
- [ ] #7 The established neighboring Jobs matrix, Ruff, compileall, Bandit, and diff hygiene pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Commit the approved design and detailed TDD implementation plan. 2. Add green public characterization coverage. 3. Add red typed-contract and direct backend-operation tests. 4. Implement transaction-neutral renewal SQL helpers and atomic backend batch operations. 5. Route the unchanged JobManager facade. 6. Run required SQLite/PostgreSQL, neighboring Jobs, lint, compile, Bandit, diff, and independent review gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved design recorded at Docs/superpowers/specs/2026-08-01-jobs-batch-lease-renewal-extraction-design.md. Work starts from fresh origin/dev f15365c7bbbcd212733551e5f56d2ed6486fffe2 on branch codex/jobs-batch-renewal-extraction. No production code changed in the design commit.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
