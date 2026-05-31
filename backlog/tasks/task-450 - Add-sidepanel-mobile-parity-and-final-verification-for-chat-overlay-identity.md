---
id: TASK-450
title: Add sidepanel mobile parity and final verification for chat overlay identity
status: Done
labels:
- implementation
- chat
- frontend
- extension
- verification
priority: high
documentation:
- Docs/superpowers/specs/2026-05-22-chat-character-overlay-and-tracked-identity-design.md
- Docs/superpowers/plans/2026-05-22-chat-character-overlay-and-tracked-identity-implementation-plan.md
modified_files:
- apps/packages/ui/src/components/Common/AssistantSelect.tsx
- apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx
- apps/packages/ui/src/components/Option/Playground/CharacterControlRail.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/CharacterControlRail.test.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/CharacterControlsSheet.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/__tests__/CharacterControlsSheet.test.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/__tests__/form.mobile-toolbar.contract.test.ts
- apps/packages/ui/src/components/Sidepanel/Chat/form.tsx
- apps/packages/ui/src/routes/sidepanel-chat-resume.ts
- apps/packages/ui/src/utils/sidepanel-overlay-resume.ts
- apps/tldw-frontend/e2e/smoke/sidepanel-nextgen-composer.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reuse the same tracked-vs-overlay assistant state contract in sidepanel/mobile chat surfaces and complete targeted verification, browser checks, and Bandit for the feature.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the follow-up review issues on sidepanel/mobile parity. The sidepanel character-controls sheet now mirrors the approved tracked-vs-overlay model: overlay actions are unavailable in tracked chats, tracked-start actions are explicit, tracked sessions are surfaced, and tracked-start clears overlay state first. AssistantSelect now refuses overlay writes when the active chat is tracked, and the sidepanel overlay-resume key contract is shared between the form and resume detector. Verification: `bunx vitest run ...` on the 11-file UI slice passed (52 tests), `bun run e2e:pw -- e2e/smoke/sidepanel-nextgen-composer.spec.ts --reporter=line` passed (5 tests), direct Chromium verification against `http://127.0.0.1:8080/__debug__/sidepanel-chat?nextgenComposer=1` confirmed the sheet opens and shows both tracked-start actions, and Bandit reported no findings while failing to parse the touched TypeScript files.
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
