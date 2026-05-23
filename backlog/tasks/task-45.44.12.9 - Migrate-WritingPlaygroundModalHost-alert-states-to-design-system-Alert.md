---
id: TASK-45.44.12.9
title: Migrate WritingPlaygroundModalHost alert states to design-system Alert
status: In Progress
labels:
- design-system
- webui
- product-state
- writing
priority: medium
parent_task_id: TASK-45.44.12
references:
- apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundModalHost.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- Docs/Design/tldw_web_design_system_contract.md
modified_files:
- apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundModalHost.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlaygroundModalHost.design-system-alert.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate WritingPlaygroundModalHost error product-state AntD Alert usages to the shared design-system Alert primitive. This reduces the Writing/Review product-state baseline while preserving modal copy and layout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The extra_body JSON error state renders through the shared design-system Alert primitive.
- [x] #2 The template-load error state renders through the shared design-system Alert primitive.
- [x] #3 The theme-load error state renders through the shared design-system Alert primitive.
- [x] #4 The modal host keeps existing copy, modal behavior, and non-alert list states unchanged.
- [x] #5 The WritingPlaygroundModalHost AntD Alert exceptions are removed from the product-state baseline.
- [x] #6 Focused tests and design-system guard verification are recorded in this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- [x] Add focused failing tests that render the extra_body JSON, template-load, and theme-load error branches and assert each uses the shared design-system Alert marker.
- [x] Replace AntD Alert usages in WritingPlaygroundModalHost with the shared design-system Alert primitive while preserving copy and layout.
- [x] Remove the migrated WritingPlaygroundModalHost rows from the product-state baseline.
- [x] Run focused tests, product-state guard/unit verifier checks, baseline JSON parse, TypeScript check, and git diff whitespace checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `WritingPlaygroundModalHost.design-system-alert.test.tsx` with focused coverage for extra_body JSON, template-load, and theme-load error branches.
- Red check before implementation: `bunx vitest run src/components/Option/WritingPlayground/__tests__/WritingPlaygroundModalHost.design-system-alert.test.tsx --reporter=dot` failed all three marker assertions because the component still rendered AntD `Alert`.
- Migrated the three modal-host AntD `Alert` usages to the shared design-system `Alert` primitive while preserving visible error copy and existing modal/list branch behavior.
- Removed the three `WritingPlaygroundModalHost` product-state baseline rows; baseline total is now 272 and Writing/Review baseline count is now 7.
- Verification:
  - `bunx vitest run src/components/Option/WritingPlayground/__tests__/WritingPlaygroundModalHost.design-system-alert.test.tsx --reporter=dot` passed, 3 tests.
  - `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed, 54 tests.
  - `bun run verify:design-system-state` passed with `Baseline exceptions: 272` and `Writing and Review surfaces: 7`.
  - Baseline parse/absence check passed and confirmed no `WritingPlaygroundModalHost` baseline entries remain.
  - `git diff --check` passed.
  - `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` exited 2 on existing repo-wide TypeScript debt; `/tmp/tldw_writing_modalhost_tsc.log` has no diagnostics for `WritingPlaygroundModalHost` or the new design-system Alert test after tightening the test `t` shim.
- Bandit not run: touched implementation is UI TypeScript/TSX and JSON/task metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
