---
id: TASK-13230
title: Make Buddy reply model recovery clear for workspace conversations
status: To Do
created_date: 2026-09-09 05:18
priority: high
references:
- TASK-13227
documentation:
- Docs/Reviews/2026-09-09-buddy-v1-qualification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During TASK-13227, a newly created workspace conversation generated successfully using the UI default model but its Buddy reply returned Choose a Chat provider and model before sending. Entering provider/model in collapsed Reply model settings recovered the reply. Clarify or repair the handoff of conversation model settings and preflight missing settings before Send.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Buddy can reuse valid model/provider settings from a newly generated workspace conversation without requiring redundant manual identifiers.
- [ ] #2 If no usable settings exist, required recovery controls and explanatory text are visible before sending and the user's draft is retained.
- [ ] #3 Provider overrides remain explicit and cannot silently redirect the target conversation or cross server/account boundaries.
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
