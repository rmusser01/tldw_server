---
id: TASK-15680
title: MFA enrolment needs the password; magic-link sign-in and unlock need an MFA
  step
status: To Do
labels:
- authnz
- security
dependencies:
- TASK-15679
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up of Q6 of the account lock plan: an intruder can enrol MFA before a lock, and magic-link sign-in and unlock carry no MFA step. Close both.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 MFA enrolment requires the current password
- [ ] #2 Magic-link sign-in and account unlock require an MFA step when the account has MFA enabled
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Renumbered from TASK-15672 on 2026-11-27: open PR #3655 took that id meanwhile.
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
