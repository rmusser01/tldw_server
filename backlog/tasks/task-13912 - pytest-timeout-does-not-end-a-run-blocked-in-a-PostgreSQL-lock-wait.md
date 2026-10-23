---
id: TASK-13912
title: pytest-timeout does not end a run blocked in a PostgreSQL lock wait
status: To Do
labels:
- tests
- postgres
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With timeout_method = thread (pyproject.toml:669) and -o timeout=150, a test whose teardown waits on a PostgreSQL lock (DROP OWNED BY behind a session idle in transaction) was not ended: measured 4:40 while planning TASK-13900 (origin/dev e06eeec68c), and 5:13, 5:37 and 13 min in TASK-13861's runs; another session's sweep was stuck 12+ min the same day. Not reproduced by the plan's reviewer (the unbounded run is forbidden). A hung CI shard waits for its job timeout instead. Find why the timer does not fire or cannot exit (psycopg's wait holding the GIL, the stack dump blocking, or another plugin), and make a hang end within the timeout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test blocked in a PostgreSQL lock wait is ended by pytest-timeout within its timeout, shown by a test or a recorded reproduction
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
