---
id: TASK-45.44.12.2
title: Migrate MediaReviewReadingPane alert to design-system Alert
status: Done
labels:
- design-system
- webui
- product-state
- review
priority: medium
parent_task_id: TASK-45.44.12
references:
- apps/packages/ui/src/components/Review/MediaReviewReadingPane.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- Docs/Design/tldw_web_design_system_contract.md
- https://github.com/rmusser01/tldw_server/pull/1964
modified_files:
- apps/packages/ui/src/components/Review/MediaReviewReadingPane.tsx
- apps/packages/ui/src/components/Review/__tests__/MediaReviewReadingPane.design-system-alert.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the single product-state AntD Alert in `MediaReviewReadingPane` to the shared design-system Alert primitive. This clears the remaining Review-area product-state baseline entry while preserving the failed-content retry affordance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] The failed-content alert in `MediaReviewReadingPane` renders through the shared design-system Alert primitive.
- [x] The alert keeps the existing failed-load copy and retry action behavior.
- [x] The `MediaReviewReadingPane` AntD Alert exception is removed from the product-state baseline.
- [x] Focused tests and design-system guard verification are recorded in this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- [x] Add a focused failing test that renders the `MediaReviewReadingPane` failed-load state and asserts the error alert uses the shared design-system Alert marker.
- [x] Replace the guarded AntD Alert usage in `MediaReviewReadingPane` with the shared design-system Alert primitive while preserving copy and action behavior.
- [x] Remove the migrated `MediaReviewReadingPane` Alert row from the product-state baseline.
- [x] Run focused tests, product-state guard/unit verifier checks, baseline JSON parse, TypeScript check, and git diff whitespace checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `MediaReviewReadingPane.design-system-alert.test.tsx`. The initial red run failed because the failed-load text was not inside `[data-ds-component="Alert"]`.
- Replaced the failed-load AntD Alert with the shared `Alert` primitive from `@/components/ui/primitives`, using `variant="error"` and the primitive action API to preserve the Retry callback.
- Removed `antd-product-state-import:src/components/Review/MediaReviewReadingPane.tsx:Alert` from the product-state baseline. Baseline count is now 290; `Writing and Review surfaces` is now 20.
- Verification:
  - `bunx vitest run src/components/Review/__tests__/MediaReviewReadingPane.design-system-alert.test.tsx --reporter=dot` passed.
  - `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed with 54 tests.
  - `node -e 'const data=JSON.parse(require("fs").readFileSync("apps/packages/ui/scripts/design-system-product-state-baseline.json","utf8")); console.log(data.length); if (data.some((entry)=>entry.path==="src/components/Review/MediaReviewReadingPane.tsx")) process.exit(1);'` printed `290`.
  - `bun run verify:design-system-state` passed and reported `Baseline exceptions: 290` and `Writing and Review surfaces: 20`.
  - `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still fails on existing repo-wide UI TypeScript debt; no diagnostics mention `MediaReviewReadingPane` or the new test.
  - `git diff --check` passed.
- Bandit skipped: touched implementation is TypeScript/React UI code and JSON/test metadata only, with no Python touched.
- PR: https://github.com/rmusser01/tldw_server/pull/1964
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the `MediaReviewReadingPane` failed-content alert to the shared design-system Alert primitive, added focused regression coverage for the design-system marker and Retry behavior, and removed the remaining Review baseline exception. PR: https://github.com/rmusser01/tldw_server/pull/1964
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
