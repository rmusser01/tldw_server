---
id: TASK-521
title: Harden chat character clear and plain-chat continuity
status: Done
labels:
- chat
- ux
- regression
- webui
priority: high
modified_files:
- apps/packages/ui/src/components/Option/Playground/Playground.tsx
- apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx
- apps/packages/ui/src/components/Option/Playground/CharacterControlRail.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/CharacterControlRail.test.tsx
- apps/packages/ui/src/store/chat-surface-coordinator.ts
- apps/packages/ui/src/store/__tests__/chat-surface-coordinator.test.ts
- apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx
- apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore reliable /chat rail behavior after the character rail removal by clearing stale tracked character assistant state and verifying the remaining plain-chat/session continuity issues on the proper chat page.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime/context clear assistant action clears visible assistant state and stale tracked server-chat metadata in /chat.
- [x] #2 Clearing assistant from a character starter state settles Runtime rail to No assistant selected and removes assistant context source.
- [x] #3 Plain chat creation helper/contract no longer returns 422 in the current real-server workflow, or stale character-control rail overlay workflow is removed/retired as superseded by rail removal.
- [x] #4 Focused unit tests and focused real-server Playwright evidence are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Cleared stale tracked assistant identity when /chat users clear the assistant from the runtime/context rails. Removed the obsolete standalone desktop CharacterControlRail component, its optional coordinator resource, its unit test, and stale real-server e2e overlay coverage. Retargeted tracked character/persona e2e coverage to the runtime rail. Verification: cockpit controls/coordinator/store/AssistantSelect Vitest suite passed (54 tests); focused real-server Playwright runtime rail flows passed (3 tests); ChatSessionCreate plain-chat schema unit tests passed (3 tests); git diff --check passed. TypeScript project check was attempted with an 8 GB heap and stopped on an unrelated existing fixture error in src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx:35. Bandit skipped because this slice touched no Python source.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
