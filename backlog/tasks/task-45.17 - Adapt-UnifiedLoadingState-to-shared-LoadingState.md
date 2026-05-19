---
id: TASK-45.17
title: Adapt UnifiedLoadingState to shared LoadingState
status: Done
assignee: []
created_date: '2026-05-08 03:05'
updated_date: '2026-05-08 03:08'
labels: []
dependencies: []
references:
  - apps/packages/ui/src/components/Common/UnifiedLoadingState.tsx
  - apps/packages/ui/src/components/ui/feedback/LoadingState.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system migration by replacing the legacy Common UnifiedLoadingState implementation with a compatibility adapter around the canonical shared LoadingState primitive while preserving existing caller behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 UnifiedLoadingState renders through the shared design-system LoadingState primitive for active loading sources.
- [x] #2 Existing children fallback, source label translation, and missing-label development diagnostics are preserved.
- [x] #3 The product-state guard baseline removes the UnifiedLoadingState local-loading-state exception without increasing other findings.
- [x] #4 Focused tests and design-system guard verification pass for the touched slice.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red coverage for LoadingState adapter conformance and UnifiedLoadingState behavior. 2. Adapt UnifiedLoadingState to the shared LoadingState primitive while preserving translated labels and children fallback. 3. Remove the obsolete baseline exception and verify the design-system guard count drops.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: bunx vitest run src/design-system/__tests__/product-state-guard.test.ts src/components/Common/__tests__/UnifiedLoadingState.test.tsx --reporter=dot passed with 43 tests. bun run verify:design-system-state passed and reports 519 baseline exceptions with local-loading-state down to 3. git diff --check passed. bunx tsc --noEmit --pretty false was attempted but still fails on pre-existing package-wide test/type errors outside this slice; no reported errors referenced the touched files. Bandit skipped because this slice only touches frontend TypeScript/JavaScript and JSON.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adapted the legacy Common UnifiedLoadingState wrapper to render through the canonical design-system LoadingState primitive, added LoadingState design-system markers for conformance tests, taught the product-state guard to allow canonical LoadingState compatibility adapters, and removed the UnifiedLoadingState local-loading-state baseline entry.
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
