---
id: TASK-45.44.12.8
title: Migrate ConnectionWebModal empty and loading states to design-system feedback
status: Done
labels:
- design-system
- webui
- product-state
- writing
priority: medium
parent_task_id: TASK-45.44.12
references:
- apps/packages/ui/src/components/Option/WritingPlayground/modals/ConnectionWebModal.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- Docs/Design/tldw_web_design_system_contract.md
- https://github.com/rmusser01/tldw_server/pull/1974
modified_files:
- apps/packages/ui/src/components/Option/WritingPlayground/modals/ConnectionWebModal.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/ConnectionWebModal.design-system-feedback.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the Connection Web modal project-required, no-data, and loading product-state AntD feedback to shared design-system feedback primitives. This reduces the Writing/Review product-state baseline while preserving modal behavior and copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The project-required empty state renders through the shared design-system EmptyState primitive.
- [x] #2 The no-data empty state renders through the shared design-system EmptyState primitive.
- [x] #3 The loading branch renders through the shared design-system LoadingState primitive.
- [x] #4 The modal keeps existing copy, query gating, and graph branch behavior.
- [x] #5 The `ConnectionWebModal` AntD Empty/Spin exceptions are removed from the product-state baseline.
- [x] #6 Focused tests and design-system guard verification are recorded in this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- [x] Add focused failing tests that render the project-required, loading, and no-data states and assert each uses the shared design-system feedback marker.
- [x] Replace the AntD Empty/Spin usages in `ConnectionWebModal` with shared design-system EmptyState/LoadingState primitives while preserving copy and layout.
- [x] Remove the migrated `ConnectionWebModal` rows from the product-state baseline.
- [x] Run focused tests, product-state guard/unit verifier checks, baseline JSON parse, TypeScript check, and git diff whitespace checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `ConnectionWebModal.design-system-feedback.test.tsx` with three product-state checks for project-required, loading, and no-data modal branches.
- Red check before implementation: `bunx vitest run src/components/Option/WritingPlayground/__tests__/ConnectionWebModal.design-system-feedback.test.tsx --reporter=dot` failed all three marker assertions because the component still rendered AntD `Empty`/`Spin`.
- Migrated the two AntD `Empty` branches to `EmptyState` and the AntD `Spin` branch to `LoadingState`, preserving the existing visible copy and conditional branches.
- Removed the three `ConnectionWebModal` product-state baseline rows; baseline total is now 280 and Writing/Review baseline count is now 10.
- Verification:
  - `bunx vitest run src/components/Option/WritingPlayground/__tests__/ConnectionWebModal.design-system-feedback.test.tsx --reporter=dot` passed, 3 tests.
  - `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed, 54 tests.
  - `bun run verify:design-system-state` passed with `Baseline exceptions: 280` and `Writing and Review surfaces: 10`.
  - Baseline parse/absence check passed and confirmed no `ConnectionWebModal` baseline entries remain.
  - `git diff --check` passed.
  - `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` exited 2 on existing repo-wide TypeScript debt; `/tmp/tldw_writing_connection_web_tsc.log` has no diagnostics for `ConnectionWebModal` or the new design-system feedback test.
- Bandit not run: touched implementation is UI TypeScript/TSX and JSON/task metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR: https://github.com/rmusser01/tldw_server/pull/1974

Migrated `ConnectionWebModal` project-required, loading, and no-data feedback branches to shared design-system `EmptyState`/`LoadingState` primitives. Added focused marker coverage and removed the three migrated product-state baseline rows, bringing the product-state baseline to 280 and Writing/Review to 10.
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
