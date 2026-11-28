---
id: TASK-15684
title: 'PostgreSQL AuthNZ tests: the app''s background pollers can recreate the pool
  under a running test'
status: To Do
labels:
- authnz
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen once while verifying the account lock (TASK-15679): the lock_db fixture's ensure_authnz_core_tables_pg returned False on the first PostgreSQL run of a session and never again. The likely cause is one of the app's background pollers recreating the AuthNZ pool on another event loop while the fixture is setting the schema up. Not reproduced since, so this task starts with finding the trigger.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The trigger is identified: which poller, and what makes it touch the pool during fixture setup
- [ ] #2 The fixture or the app's test mode keeps pollers off the pool during a test
- [ ] #3 The PostgreSQL account-lock tests pass 20 consecutive cold runs
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
