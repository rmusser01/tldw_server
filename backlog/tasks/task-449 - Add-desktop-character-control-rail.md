---
id: TASK-449
title: Add desktop character control rail
status: Done
labels:
- implementation
- chat
- frontend
- ui
priority: high
documentation:
- Docs/superpowers/specs/2026-05-22-chat-character-overlay-and-tracked-identity-design.md
- Docs/superpowers/plans/2026-05-22-chat-character-overlay-and-tracked-identity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the main /chat desktop character/persona control rail as an additive surface for overlay and tracked-session actions without replacing the existing chat UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the desktop character control rail as an additive right-side `/chat` surface. The rail now shows the current plain/overlay/tracked mode summary, separates overlay actions from tracked-start actions, exposes tracked-session open actions, and mounts through the chat surface coordinator without replacing the existing transcript/composer UI. The coordinator gained a `character-control` panel id, `PlaygroundForm` now enables that panel on desktop only, and `Playground` renders the rail when the coordinator marks it visible. Verification: `node node_modules/vitest/vitest.mjs --config vitest.config.ts run src/components/Option/Playground/__tests__/CharacterControlRail.test.tsx src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx src/store/__tests__/chat-surface-coordinator.test.ts` -> 3 files passed, 10 tests passed. Bandit not applicable because the touched scope in this task is TypeScript/TSX only.
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
