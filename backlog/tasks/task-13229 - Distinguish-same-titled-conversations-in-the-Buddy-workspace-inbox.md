---
id: TASK-13229
title: Distinguish same-titled conversations in the Buddy workspace inbox
status: To Do
created_date: 2026-09-09 05:16
priority: medium
references:
- TASK-13227
documentation:
- Docs/Reviews/2026-09-09-buddy-v1-qualification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13227 created two workspace conversations with the same generated title during send/retry. Buddy correctly stores separate IDs but renders identical selector and result labels, making the user's reply target hard to distinguish. Provide concise visible and accessible distinguishing context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user can distinguish same-titled conversations in both the workspace Buddy selector and result list before replying or acknowledging.
- [ ] #2 Visible and accessible distinguishing context consistently identifies the selected conversation without replacing its saved title.
- [ ] #3 Reply and acknowledgement remain bound to the explicit stable conversation ID; different-title labels remain concise.
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
