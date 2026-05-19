---
id: TASK-45.17.2
title: Tighten LoadingState adapter guard return detection
status: Done
assignee: []
created_date: '2026-05-08 05:47'
updated_date: '2026-05-08 05:49'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1374'
  - apps/packages/ui/scripts/design-system-product-state-rules.mjs
  - apps/packages/ui/src/design-system/__tests__/product-state-guard.test.ts
parent_task_id: TASK-45.17
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address PR 1374 review feedback that the product-state guard currently treats any nested LoadingState JSX as a canonical loading adapter. Tighten the adapter allowance so local loading-state components are exempt only when their returned loading UI is the canonical LoadingState primitive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Guard allows direct LoadingState return adapters, including simple conditional return expressions.
- [x] #2 Guard still flags local loading components that merely nest LoadingState while returning bespoke loading UI.
- [x] #3 Focused product-state guard tests and design-system verifier pass after the rule change.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red guard coverage for nested-only LoadingState usage and conditional direct returns. 2. Replace loading adapter owner detection with direct return-expression detection while leaving EmptyState detection unchanged. 3. Rerun focused guard tests and the full design-system state verifier.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot failed first because nested LoadingState usage produced no local-loading-state finding. After the rule change, bunx vitest run src/components/Common/__tests__/UnifiedLoadingState.test.tsx src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed with 46 tests. bun run verify:design-system-state passed and baseline exceptions remained 519 with local-loading-state at 3. git diff --check passed. Bandit skipped because this is frontend JavaScript/test-only guard work.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Tightened LoadingState adapter guard detection so local loading wrappers are exempt only when they directly return the canonical LoadingState primitive, including conditional returns, while nested-only LoadingState usage remains flagged.
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
