---
id: TASK-12160
title: Fix PostgreSQL setup self-verify timestamp timezone mismatch
status: Done
labels:
- bug
- postgres
- authnz
references:
- https://github.com/rmusser01/tldw_server/issues/2651
modified_files:
- tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py
- tldw_Server_API/app/services/startup_auth.py
- tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_user_timestamps.py
- tldw_Server_API/tests/AuthNZ_Postgres/test_user_timestamp_timezones_pg.py
- tldw_Server_API/tests/Services/test_startup_auth.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix GitHub issue #2651: first-time setup self-verification fails on PostgreSQL with asyncpg DataError "can't subtract offset-naive and offset-aware datetimes" when marking the account verified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Root cause documented against the setup self-verify/auth service path.
- [ ] #2 Latest dev checked for whether the issue is already fixed.
- [ ] #3 Regression test covers aware UTC datetimes being written through mark_user_verified against PostgreSQL timestamp columns or normalized parameters.
- [ ] #4 Minimal pragmatic fix implemented without broad schema churn.
- [ ] #5 Targeted tests, Bandit on touched scope, and relevant verification commands recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Root cause: setup self-verify passes an aware UTC datetime into auth_service.mark_user_verified, but legacy PostgreSQL users.updated_at columns may be timestamp without time zone, which asyncpg rejects. Latest dev had new embedded bootstrap schemas using TIMESTAMPTZ, but no startup repair for existing/fixture schemas, so the issue was not fully fixed. Fix: add an idempotent PostgreSQL startup ensure that converts legacy users timestamp columns to TIMESTAMPTZ using UTC, and wire it before other PG AuthNZ extras. Tests: unit coverage for the new migration SQL and startup wiring, plus a PostgreSQL regression that starts from a TIMESTAMP users.updated_at column and verifies mark_user_verified accepts an aware UTC datetime after repair. Verification: targeted unit suite passed, forced PostgreSQL regression passed, production-code Bandit passed; full touched-scope Bandit only reported pytest assert B101 findings in tests.
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
