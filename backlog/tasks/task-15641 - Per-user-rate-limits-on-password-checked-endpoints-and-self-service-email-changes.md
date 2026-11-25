---
id: TASK-15641
title: Per-user rate limits on password-checked endpoints and self-service email changes
status: To Do
labels:
- authnz
- security
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A wrong current password is answered without any per-user limit on PUT /api/v1/users/me, PATCH /api/v1/users/me/profile (email change, TASK-15640) and POST /api/v1/users/change-password, so a signed-in session can guess the password. Successful self-service email changes are not limited either, and each one mails the previous address: an account can set its email to someone else's address and change it away again, repeatedly, to flood that inbox. Add a per-user limit on failed password checks shared by these endpoints, and a per-user cap on successful self-service email changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Repeated wrong current passwords on these endpoints are refused after a per-user limit, on SQLite and PostgreSQL
- [ ] #2 The limit does not lock the account or affect sign-in
- [ ] #3 A per-user cap on successful self-service email changes (e.g. 5 per 24 h) on PUT /users/me and PATCH v1/v2 users/me/profile
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
