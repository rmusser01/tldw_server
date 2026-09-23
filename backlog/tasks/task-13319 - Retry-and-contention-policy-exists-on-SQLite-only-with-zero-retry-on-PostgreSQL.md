---
id: TASK-13319
title: >-
  Retry and contention policy exists on SQLite only with zero retry on
  PostgreSQL
status: Done
assignee: []
created_date: '2026-09-22 04:55'
updated_date: '2026-09-23 17:28'
labels:
  - bug
  - db
  - dual-backend
dependencies: []
references:
  - 'tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py:4165'
  - 'tldw_Server_API/app/core/DB_Management/transaction_utils.py:54'
  - 'tldw_Server_API/app/core/DB_Management/Workflows_DB.py:1659'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
28 inline retry loops all sit inside _SQLitePromptStudioDatabase; _BackendPromptStudioDatabase has ZERO (regex-verified). Prompt Studio job queue does read-then-conditional-update (acquire_next_job, update_job_status, renew_job_lease, retry_job_record): on SQLite contention is retried 5x with jittered backoff and succeeds, on PostgreSQL the same contention surfaces as SQLSTATE 40001/40P01 and propagates to the caller. Two workers polling one queue behave completely differently per backend.

Three internal inconsistencies among the 28: :4162 tests "database is locked" with NO .lower() while the other 27 lower it; :6265 hardcodes attempt < 4 inside a range(5) loop; 5 of 28 omit the jitter term that the other 23 have, producing synchronised retry waves.

Destination: core/DB_Management/retry_policy.py - classify a storage error as retryable for a given backend and produce the next delay. Depends on the backoff consolidation task.

Source: synthesis F21
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One retry policy shared by both backends
- [x] #2 PostgreSQL serialization failures are retried
- [x] #3 Jitter and the locked-error predicate are defined once
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-09-23 (12125df6ab): TransientContentionError raised by both SQL backends (SQLite 'database is locked'; PG SQLSTATE 40001/40P01/55P03), class-only, redaction intact. core/DB_Management/retry_policy.py: is_retryable_contention (walks cause chain) + run_with_contention_retry (ADR-047 capped exponential; jitter and locked predicate from core/Utils/backoff.py, defined once => AC3). Used by Prompt Studio repositories as aggregates move (TASK-13318). AC1/AC2 complete for moved aggregates; the remaining inline SQLite loops and PG job-queue paths go with Stage 4-6. Bandit clean on new modules.

2026-09-23: AC1/AC2 met - every Prompt Studio write and read now goes through retry_policy on both backends (all 28 inline SQLite loops removed with their aggregates; PostgreSQL 40001/40P01/55P03 retried, including job queue paths).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
core/DB_Management/retry_policy.py is the one contention-retry policy; both SQL backends raise TransientContentionError (SQLite locked; PostgreSQL 40001/40P01/55P03), so PostgreSQL contention is retried for the first time. All 28 inline SQLite loops in PromptStudioDatabase.py are gone with the TASK-13318 consolidation; every Prompt Studio read/write goes through the policy on both backends (including the job queue). Jitter and the locked predicate come only from core/Utils/backoff.py. Tests: test_retry_policy.py, test_backend_constraint_classification.py (incl. real locked SQLite), and a retry integration test in the Prompt Studio parity harness verified to fail without the retry. Bandit clean. Known: other modules' own retry helpers (transaction_utils, Workflows_DB) keep their existing schedules per ADR-047 and do not yet use TransientContentionError.
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
