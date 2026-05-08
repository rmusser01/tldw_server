---
id: TASK-45.17.1
title: Address PR 1374 UnifiedLoadingState review comments
status: Done
assignee: []
created_date: '2026-05-08 05:35'
updated_date: '2026-05-08 05:36'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1374'
  - apps/packages/ui/src/components/Common/UnifiedLoadingState.tsx
  - >-
    apps/packages/ui/src/components/Common/__tests__/UnifiedLoadingState.test.tsx
parent_task_id: TASK-45.17
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the actionable PR 1374 review feedback on the UnifiedLoadingState design-system adapter by making the loaded-state compatibility boundary explicit and documenting it in focused tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 UnifiedLoadingState returns children without the LoadingState wrapper when no sources are loading.
- [x] #2 Label translation for active loading sources only performs fallback translation when labels are shown.
- [x] #3 Focused UnifiedLoadingState and product-state guard tests pass after the review fix.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused review-regression coverage for the loaded state and hidden labels. 2. Restore the explicit no-loading fragment return while keeping hooks unconditional. 3. Skip translation fallback work when source labels are hidden and rerun focused verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review comments addressed: restored explicit children-only return when loadingSources is empty; added a test that no LoadingState marker exists after loading; added a red-green test proving hidden labels do not call the translation fallback; kept product-state guard behavior unchanged. Verification: bunx vitest run src/components/Common/__tests__/UnifiedLoadingState.test.tsx src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed with 44 tests. bun run verify:design-system-state passed with baseline exceptions still at 519 and local-loading-state at 3. git diff --check passed. Bandit skipped because this is frontend TypeScript/test-only work.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR 1374 UnifiedLoadingState review comments by making the loaded-state fragment return explicit, avoiding hidden-label translation work, and extending tests to prevent the wrapper/layout regression.
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
