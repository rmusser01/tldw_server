---
id: TASK-45.51
title: Migrate RecentStudySessions empty states to design-system primitives
status: Done
labels:
- design-system
- product-state
- ui
- flashcards
parent_task_id: TASK-45
references:
- apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx
- apps/packages/ui/src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system product-state migration by replacing RecentStudySessions AntD Empty product-state surfaces with canonical design-system EmptyState/LoadingState primitives while preserving the existing loading, retryable error, no-sessions, and session-list behaviors. Remove the matching product-state baseline exceptions and verify the guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Verification: `bunx vitest run src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx --reporter=dot` passed (3 tests); `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed (52 tests); `bun run verify:design-system-state` passed with baseline exceptions now 323; `git diff --check` passed. `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still fails on inherited repo-wide TypeScript debt unrelated to this slice. Bandit skipped because no Python was touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated RecentStudySessions loading, retryable error, and no-session product states from AntD Empty to canonical design-system LoadingState/EmptyState primitives. Added focused DOM coverage that asserts those states render through data-ds-component markers, removed the two matching RecentStudySessions product-state baseline exceptions, and verified the focused component test, product-state guard unit test, design-system state verifier, and git diff whitespace. Bandit was not run because this slice touched only TypeScript/TSX UI, JSON baseline, and Backlog metadata. Full TypeScript still fails on inherited repo-wide debt unrelated to this slice; no diagnostics referenced the changed RecentStudySessions files.
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
