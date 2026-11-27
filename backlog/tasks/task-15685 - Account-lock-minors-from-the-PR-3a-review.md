---
id: TASK-15685
title: 'Account lock: minors from the PR 3a review'
status: To Do
labels:
- authnz
- security
dependencies:
- TASK-15679
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Minor findings of the independent review of PR #3664 (TASK-15679) that were not fixed there. None is exploitable on its own beyond what ADR 104 already lists as a known limit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The take-back's credentials_valid_after is past its own final second, so a session-less token minted in that second by a request that passed its check before the forced lock does not survive (ADR 104 known limit removed)
- [ ] #2 Registration, and magic-link auto-registration, refuse an address another account released in the last 7 days, without telling an anonymous caller why
- [ ] #3 find_locked_user_by_previous_email no longer scans the users table on every anonymous magic-link request (index on both backends, or a cheaper lookup)
- [ ] #4 No lock-mail audit event is written when no mail was sent, or the event says so in a way the security page can show
- [ ] #5 An org key and a team key owned by a locked account are each refused, proven by a test per key kind
- [ ] #6 A chat actor id made of non-ASCII digits is not read as a user id (locked_actor_error uses an ASCII check), proven by a test
- [ ] #7 A take-back counts its look-back from the earliest email change inside the window, not the latest, or ADR 104 says why not
- [ ] #8 A cancelled request cannot re-lock a take-back that already won its compare-and-set (the wait for credentials_valid_after moves out of the guarded block)
- [ ] #9 The take-back hashes its replacement password before it deletes the SSO links, so the hashing time is not added to the window between the delete and the compare-and-set
- [ ] #10 The MFA sign-in test for a locked account fails when the route's own lock check is removed (assert the challenge is not consumed or the code is never verified)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the re-review of the PR 3a fixes (2026-11-27): a persistent error deleting SSO links other than a missing PostgreSQL table (for example insufficient privilege) now blocks every take-back while the newer address can still unlock plainly; decide whether that should alert an administrator. Also: forgot-password response time already tells a known address from an unknown one on a healthy install (one SMTP send against none), which predates the lock.
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
