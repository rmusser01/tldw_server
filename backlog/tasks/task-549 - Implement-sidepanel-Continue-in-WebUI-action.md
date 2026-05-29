---
id: TASK-549
title: Implement sidepanel Continue in WebUI action
status: To Do
labels:
- chat
- extension
- implementation
priority: Medium
references:
- TASK-546
- TASK-547
- TASK-548
documentation:
- Docs/superpowers/specs/2026-05-29-sidepanel-chat-webui-handoff-design.md
- Docs/superpowers/plans/2026-05-29-sidepanel-chat-webui-handoff-implementation.md
modified_files:
- apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/form.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ControlRow.chat-handoff.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 2 from the sidepanel chat WebUI handoff plan: add a sidepanel ControlRow Continue in WebUI action that creates a handoff package, opens /chat with the handoff id, preserves the existing route-only full-app action, passes draft and visible page context from the sidepanel form, and covers the flow with focused regression tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
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
