---
id: TASK-13911
title: PostgreSQL test fixtures hand-roll a restricted-role teardown that waits forever
  on a leaked session
status: To Do
labels:
- tests
- postgres
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
34 test files (git grep -l 'DROP OWNED BY' -- tldw_Server_API/tests, minus tests/_plugins/postgres.py and the RAG file TASK-13900 fixed; measured on 617094d014) create a restricted role and drop it with a bare DROP OWNED BY. A session the test leaks in a transaction makes that DROP wait for its lock forever: TASK-13900's RAG fixture waited past pytest-timeout (150 s) for 4:40 and more, and left its role in the shared cluster. pg_restricted_backend and _drop_restricted_role (tests/_plugins/postgres.py, TASK-13861) fail fast and name the session. plan_fu2_claims.md adds another hand-rolled fixture with only a lock_timeout. Move these fixtures to pg_restricted_backend or _drop_restricted_role (consider making it public).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No test file drops a restricted role with a bare DROP OWNED BY: each uses pg_restricted_backend or the shared fail-fast drop, with a guard test that greps for the pattern
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
