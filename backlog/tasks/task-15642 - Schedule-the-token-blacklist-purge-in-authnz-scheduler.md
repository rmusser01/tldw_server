---
id: TASK-15642
title: Schedule the token_blacklist purge in authnz_scheduler
status: To Do
labels:
- authnz
- security
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TokenBlacklist.cleanup_expired (token_blacklist.py:535) has no caller, so token_blacklist grows without bound; the session cap (TASK-15628) writes two rows per evicted JWT session. Schedule it in services/authnz_scheduler.py next to the session cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The purge runs on the authnz scheduler on SQLite and PostgreSQL
- [ ] #2 Rows are deleted only after their expires_at
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
