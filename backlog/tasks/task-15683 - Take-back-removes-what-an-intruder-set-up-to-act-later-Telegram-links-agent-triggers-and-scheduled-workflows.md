---
id: TASK-15683
title: 'Take-back removes what an intruder set up to act later: Telegram links, agent
  triggers and scheduled workflows'
status: To Do
labels:
- authnz
- security
dependencies:
- TASK-15679
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An account take-back (TASK-15679) restores the previous address, voids the password, ends sessions, suspends recent API keys and removes recent SSO links. It leaves in place the things an intruder could have set up to keep acting for the account without a credential: Telegram links, ACP agent triggers and scheduled workflows. They are refused while the account is locked but act again after the unlock. Build with the background pause (PR 4 of the session feature), which already has to find this work in order to pause it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On a take-back, Telegram links, agent triggers and scheduled workflows created inside the take-back window are removed or disabled, proven by tests on SQLite and PostgreSQL
- [ ] #2 The owner is told what was removed (unlock mail or the security page) and the removal is audited
- [ ] #3 Whether a plain unlock does the same is decided with the owner and recorded in ADR 104
- [ ] #4 ADR 104 known limits are updated to match
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
