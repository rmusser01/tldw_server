---
id: TASK-15632
title: Session cap, new-sign-in alert and account lock
status: To Do
labels:
- authnz
- security
- webui
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Limit each user to 4 signed-in sessions, alert a user's other sessions when a new one is created, and let the user lock the account (sign out everything, suspend API keys, re-entry only by emailed link). Where no email can be sent the alert offers sign-out-others instead of lock. Motivated by the unbounded single-user cookie session minting found on 2026-11-23 (TASK-15628) and by account-takeover response. This task covers the design; implementation ships as three PRs (cap; alert + sessions page; lock + email link).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design doc approved by the owner
- [ ] #2 ADR drafted for the session cap and account lock rules
- [ ] #3 Implementation plan written from the approved design
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
