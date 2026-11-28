---
id: TASK-15682
title: 'SQLite: routes that hold a request transaction and write through a second
  connection wait on their own write lock'
status: To Do
labels:
- authnz
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A route that takes Depends(get_db_transaction) holds SQLite's write lock (BEGIN IMMEDIATE) for the whole request. Anything it then writes through another pool connection waits on that lock until busy_timeout, then fails or is swallowed. Found while building the account lock (TASK-15679): reset-password revoked sessions that way and answered 500, or 200 with every session still valid (fixed there by revoking after the commit). The same class remains elsewhere: the admin step-up's admin_reauth_token is not consumed on SQLite, and revoke_all_user_tokens swallows the error and returns 0, so callers cannot tell a failed revoke from nothing to revoke.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every route that takes get_db_transaction and writes through a second connection is listed, with the write it makes
- [ ] #2 The admin step-up token is consumed on SQLite, proven by a test that replays it
- [ ] #3 revoke_all_user_tokens tells its caller when the revoke failed
- [ ] #4 Each listed route is fixed, or its write moved after the commit, with a SQLite test that fails on the old behaviour
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
