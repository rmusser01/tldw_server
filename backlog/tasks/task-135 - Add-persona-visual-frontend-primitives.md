---
id: TASK-135
title: Add persona visual frontend primitives
status: Done
assignee: []
created_date: '2026-05-09 00:30'
updated_date: '2026-05-09 00:37'
labels:
  - persona
  - webui
  - frontend
  - implementation
dependencies:
  - TASK-132
documentation:
  - Docs/superpowers/specs/2026-05-08-persona-visual-packs-design.md
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add frontend type contracts, API client helpers, visual-state resolver, and sprite-frame renderer primitives for persona visual packs. Keep shell integration and editor UX for later plan slices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frontend persona visual pack, asset, manifest, animation, and runtime state types are defined
- [x] #2 API helper functions wrap persona visual pack list/detail/create/update/upload/activate/deactivate flows
- [x] #3 Visual state resolver maps live voice/tool/wake/error/override trigger inputs with documented priority
- [x] #4 SpriteFrameRenderer renders image frames, preview frames, ordered animations, sprite-sheet regions, idle fallback, data-visual-state, and render errors
- [x] #5 Focused Vitest coverage exercises resolver priority and renderer behavior
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Environment setup: initial `bunx vitest run src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx` could not load local `vitest/config` because the workspace dependencies were not hydrated. Ran `bun install --frozen-lockfile` from `apps/`; lockfile was unchanged.

Red verification after dependency hydration: `bunx vitest run src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx` failed because `../personaVisualState` and `../SpriteFrameRenderer` did not exist.

Green verification: `bunx vitest run src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx` passed, 14 passed.

Static check note: full UI `tsc --noEmit` still reports pre-existing unrelated type errors across the package. A filtered rerun found no errors referencing the touched persona visual files after fixing the persona visual service payload typing.

Hygiene/security: `git diff --check` passed. Bandit is not applicable because this slice only touches TypeScript/React frontend files and markdown task tracking.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added frontend persona visual contracts, API helpers for visual-pack CRUD/upload/activation/review flows, a priority-based visual-state resolver, and a sprite-frame renderer that supports preview frames, explicit frame order, sprite-sheet regions, idle fallback, and render-error callbacks. Focused Vitest coverage locks the resolver and renderer behavior for the next shell-integration slice.
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
