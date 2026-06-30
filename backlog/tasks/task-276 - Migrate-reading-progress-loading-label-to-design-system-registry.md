---
id: TASK-276
title: Migrate reading progress loading label to design-system registry
status: Done
assignee: []
created_date: '2026-05-12 00:18'
labels:
  - design-system
  - frontend
  - product-state
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the remaining hardcoded reading-progress loading state label from the product-state guard baseline by routing it through the design-system state registry without changing reading progress behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The reading progress loading label is sourced from the design-system state registry instead of a hardcoded canonical label.
- [x] #2 The product-state guard baseline no longer includes the reading progress loading-label exception.
- [x] #3 Focused regression coverage proves the registry fallback is used for the loading label.
- [x] #4 Design-system guard verification passes for this slice.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

- Updated `useReadingProgress` so the loading live-region prefix uses `getDesignSystemState('loading').label` as the i18n default value.
- Updated the stage 15 accessibility regression test to mock the design-system loading label and assert that registry fallback appears in the content-selection announcement.
- Removed the matching `canonical-state-label` baseline exception for `src/components/Media/hooks/useReadingProgress.tsx`.

## Verification

- PR: https://github.com/rmusser01/tldw_server/pull/1577
- `bunx vitest run src/components/Media/__tests__/ContentViewer.stage15.accessibility.test.tsx --reporter=dot`
- `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot`
- `bun run verify:design-system-state`
- `git diff --check`
- `bunx tsc --noEmit --pretty false 2>&1 | rg -n "useReadingProgress|ContentViewer.stage15.accessibility|design-system-product-state-baseline"` returned no touched-path matches; full frontend `tsc` remains repo-noisy.
- Bandit skipped: frontend TypeScript/test-only slice with no Python runtime changes.

## Final Summary

Migrated the reading-progress loading announcement label to the design-system state registry and removed its product-state guard baseline exception. The focused accessibility test now proves the loading prefix follows the registry fallback while preserving the ready announcement behavior.
