---
id: TASK-13672
title: Preserve native trigger and accessibility props in the shared button
status: Done
labels:
- chat
- ux
- accessibility
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The shared Button drops standard native hover, focus, pointer, accessibility and form props, including props cloned by installed AntD tooltip/popover triggers. Extend the existing button contract and forwarding once, preserving custom variant and legacy type compatibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual installed tooltip hover and keyboard-focus triggers work through the shared Button.
- [x] #2 Native accessibility, form attributes, events and refs reach the button without leaking custom styling props.
- [x] #3 Connected caller tests, dependency types, lint, bounded review and source-matched build pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new durable architectural rule; use native React button attributes and existing installed AntD. Distinct from the already fixed Tools pointer layout shift (TASK-13669).
Final bounded evidence: Docs/Reviews/CHAT_FRESH_SAVED_READINESS_NATIVE_BUTTON_13671_13672_2026_10_14.md. Relevant connected tests, actual installed AntD controls, both production type profiles, migrated private fixture dependency types, source-matched production assets/tokens/budgets and independent source review pass. Native and WebUI actual private protected protocol passes with exact bytes/IDs and no live setup dependency. Original9conversations/28rows equal immutable baseline. TS/JSON/Markdown scope: Bandit not applicable. Separate master early-Stop/acceptance matrix and final whole-PR gates remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Forwarded native button trigger/accessibility/form props using existing React attributes, preserving legacy styling/type/loading/ref behavior; installed Tooltip hover/focus and connected callers qualify.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
