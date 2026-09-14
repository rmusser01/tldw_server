---
id: TASK-13010
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
updated_date: 2026-08-10 19:38
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Characterize and extract JobManager.batch_renew_leases into dedicated SQLite and PostgreSQL lifecycle operations while preserving the public facade, input-order processing, exact row-count semantics, backend-specific clock timing, expected no-op behavior, and one atomic scope for the complete batch.
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
Approved design recorded at Docs/superpowers/specs/2026-08-01-jobs-batch-lease-renewal-extraction-design.md. Work started on branch codex/jobs-batch-renewal-extraction and was rebased cleanly onto origin/dev 45490da82e, then rebased cleanly again onto origin/dev 7b48bcb04f for PR #2773 follow-up. The provisional TASK-12989 identity collided with two records already on dev; independent review validated the tooling problem and this Jobs task was manually renumbered to unique TASK-13010 under the requester's prior approval for narrowly scoped repair of this record. Qodo follow-up validation rejected the fixture-isolation claim because jobs_pg_dsn already depends on the canonical function-scoped pg_temp_db fixture. The validated SQL-construction concern is addressed with psycopg.sql composition for PostgreSQL and fixed SQLite trigger SQL keyed to a constant test identity. The validated SQLite transaction concern is addressed with a savepoint for direct calls inside caller-owned transactions. Independent review then found that a SQLite whole-transaction abort could let cleanup mask the primary error; the final helper preserves the primary error and chains cleanup failure, with RAISE(ROLLBACK) regression coverage. The review also confirmed complete normalization before clock, cursor/RLS, and backend dispatch is the approved behavior; explicit SQLite/PostgreSQL routing coverage and compatibility documentation now lock that precedence. Final post-rebase verification passed 97 focused SQLite/contracts tests, 43 required PostgreSQL/RLS tests with zero skips, 50 neighboring lifecycle/parity tests with 13 PostgreSQL-only skips confined to the non-required SQLite matrix, and 11 required PostgreSQL parity tests with zero skips. Ruff, compileall, diff hygiene, unique-task-ID, and operation/manager boundary checks passed. Bandit reported zero findings across 9,290 lines in /tmp/bandit_task_12989_post_rebase_qodo.json and zero findings for the final 432-line SQLite scope in /tmp/bandit_task_13010_final.json. Independent remediation re-review found no remaining code, test, documentation, or tracker issue.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Extracted atomic batch lease renewal into dedicated SQLite and PostgreSQL lifecycle operations while preserving the JobManager public contract, ordered duplicate attempts, expected no-ops, non-shortening leases, duration clamping, backend-specific clock behavior, PostgreSQL RLS cursor setup, and one atomic scope per batch. Added immutable contracts plus public characterization, direct backend, routing, property, real-database rollback, caller-owned savepoint, and primary-error-precedence coverage. The branch is rebased onto dev 7b48bcb04f and local verification and independent review are clean. Pull request: https://github.com/rmusser01/tldw_server/pull/2773. Merge remains blocked until the requester replaces the pending PR Summary with their required human-authored Change summary; this tracker summary is verification metadata and does not satisfy that policy.
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
