---
id: TASK-13393
title: AuthNZ Postgres startup loses the profile_version corruption reason
status: Done
assignee: []
created_date: '2026-09-28 03:25'
updated_date: '2026-09-30 04:28'
labels:
  - authnz
  - postgres
  - observability
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ensure_authnz_core_tables_pg (pg_migrations_extra.py) runs ensure_postgres_profile_version_on_connection inside db_pool.transaction(). A corrupt profile_version (e.g. NULL) raises a RuntimeError naming profile_version, but the transaction boundary sanitizes it to TransactionError and the noncritical except returns False, logging only 'Failed to ensure PostgreSQL AuthNZ core tables' with exception_type=TransactionError. Startup still fails closed (initialize.py raises 'Failed to ensure Postgres AuthNZ core tables'), but the operator never learns the cause. Regressed in 5f31630280; test_postgres_current_schema_corruption_fails_closed_at_startup was relaxed to assert False when the full suite was re-enabled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A corrupt profile_version at startup surfaces a message naming profile_version to the operator (log or raised error), without leaking row data
- [x] #2 test_postgres_current_schema_corruption_fails_closed_at_startup asserts that reason
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the PostgreSQL transaction boundary (AuthNZ/database.py) re-raises every failure as TransactionError('PostgreSQL transaction') and logs only the exception type chain, because database messages can carry row data. The profile_version readiness RuntimeError lost its fixed-text reason there. Fix: readiness checks in postgres_profile_version_schema.py and profile_candidate_schema.py raise SchemaReadinessError (RuntimeError subclass, AuthNZ/exceptions.py); the boundary passes only that type's message as TransactionError detail; ensure_authnz_core_tables_pg logs the sanitized TransactionError message, and pool-init schema readiness logs the reason. Other failures stay type-only. Test: test_postgres_current_schema_corruption_fails_closed_at_startup now captures the WARNING and asserts it names 'AuthNZ profile_version readiness validation failed'. Verified: test_profile_version_migration_pg.py 8 passed (Docker Postgres); 14 AuthNZ/DB test files touching these modules 271 passed. Bandit (-ll) on the 5 touched modules: 0 issues. Docs: none needed (operator log text only). No known skips.

Qodo on #3063: SchemaReadinessError moved to app/core/exceptions.py (the repo rule). test_profile_version_migration_pg.py 8 passed; the AuthNZ/DB files touching these modules 151 passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A corrupt users.profile_version at PostgreSQL startup now logs 'Failed to ensure PostgreSQL AuthNZ core tables: Transaction failed during: PostgreSQL transaction - AuthNZ profile_version readiness validation failed' rather than a bare TransactionError. Readiness checks raise SchemaReadinessError (fixed text, core exceptions), and only that message crosses the transaction boundary's sanitizer. The startup test asserts the reason.
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
