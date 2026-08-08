---
id: TASK-12989
title: Extract Jobs batch lease renewal operations atomically
status: Done
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
- https://github.com/rmusser01/tldw_server/pull/2773
documentation:
- Docs/superpowers/specs/2026-08-01-jobs-batch-lease-renewal-extraction-design.md
- Docs/superpowers/plans/2026-08-08-jobs-batch-lease-renewal-extraction.md
modified_files:
- tldw_Server_API/app/core/Jobs/manager.py
- tldw_Server_API/app/core/Jobs/operations/contracts.py
- tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py
- tldw_Server_API/app/core/Jobs/operations/postgres/lifecycle.py
- tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_sqlite.py
- tldw_Server_API/tests/Jobs/test_jobs_batch_renew_characterization_postgres.py
- tldw_Server_API/tests/Jobs/test_jobs_batch_renew_routing.py
- tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py
- tldw_Server_API/tests/Jobs/property/test_operation_contract_properties.py
- tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_sqlite.py
- tldw_Server_API/tests/Jobs/test_jobs_renew_release_operations_postgres.py
updated_date: 2026-08-08 14:36
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Characterize and extract JobManager.batch_renew_leases into dedicated SQLite and PostgreSQL lifecycle operations while preserving the public facade, input-order processing, exact row-count semantics, backend-specific clock timing, expected no-op behavior, and one atomic transaction for the complete batch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 JobManager.batch_renew_leases keeps its public signature and integer return contract.
- [x] #2 SQLite and PostgreSQL batch renewal route through dedicated backend lifecycle operations that do not import JobManager.
- [x] #3 Unexpected backend or clock failures roll back every earlier renewal in the batch.
- [x] #4 Missing, non-processing, and stale-lease items remain non-fatal no-ops while valid items commit.
- [x] #5 Duration clamping, duplicate-item counting, empty-batch behavior, and backend-specific clock timing remain compatible.
- [x] #6 Focused SQLite and required real PostgreSQL suites pass with zero PostgreSQL skips.
- [x] #7 The established neighboring Jobs matrix, Ruff, compileall, Bandit, and diff hygiene pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Commit the approved design and detailed TDD implementation plan. 2. Add green public characterization coverage. 3. Add red typed-contract and direct backend-operation tests. 4. Implement transaction-neutral renewal SQL helpers and atomic backend batch operations. 5. Route the unchanged JobManager facade. 6. Run required SQLite/PostgreSQL, neighboring Jobs, lint, compile, Bandit, diff, and independent review gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved design recorded at Docs/superpowers/specs/2026-08-01-jobs-batch-lease-renewal-extraction-design.md. Work started on branch codex/jobs-batch-renewal-extraction and was rebased cleanly onto origin/dev 45490da82e. Post-rebase verification at 2c6a987b40 passed 93 focused SQLite/contracts tests, 43 required PostgreSQL/RLS tests with zero skips, 50 neighboring lifecycle/parity tests with 13 PostgreSQL-only skips confined to the non-required SQLite matrix, and 11 required PostgreSQL parity tests with zero skips. Ruff, compileall, diff hygiene, and operation/manager boundary checks passed. Bandit reported zero findings across 9,273 lines in /tmp/bandit_task_12989_post_rebase.json. Independent task and whole-branch reviews found no remaining code issues.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Extracted atomic batch lease renewal into dedicated SQLite and PostgreSQL lifecycle operations while preserving the JobManager public contract, ordered duplicate attempts, expected no-ops, non-shortening leases, duration clamping, backend-specific clock behavior, PostgreSQL RLS cursor setup, and one transaction per batch. Added immutable contracts plus public characterization, direct backend, routing, property, and real-database rollback coverage. The rebased branch is 0 behind current dev and technically review-clean. Draft pull request: https://github.com/rmusser01/tldw_server/pull/2773. Merge remains blocked until the requester adds the required human-authored Change summary to the pull request; this tracker summary is verification metadata and does not satisfy that policy.
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
