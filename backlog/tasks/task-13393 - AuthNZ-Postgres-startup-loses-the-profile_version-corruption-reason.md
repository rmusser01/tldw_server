---
id: TASK-13393
title: AuthNZ Postgres startup loses the profile_version corruption reason
status: To Do
assignee: []
created_date: '2026-09-28 03:25'
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
- [ ] #1 A corrupt profile_version at startup surfaces a message naming profile_version to the operator (log or raised error), without leaking row data
- [ ] #2 test_postgres_current_schema_corruption_fails_closed_at_startup asserts that reason
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
