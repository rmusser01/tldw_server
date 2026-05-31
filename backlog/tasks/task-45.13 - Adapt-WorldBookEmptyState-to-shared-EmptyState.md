---
id: TASK-45.13
title: Adapt WorldBookEmptyState to shared EmptyState
status: Done
assignee: []
created_date: '2026-05-07 01:35'
updated_date: '2026-05-07 01:46'
labels:
  - design-system
  - frontend
  - worldbooks
dependencies: []
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
  - Docs/Design/tldw_web_design_system_inventory.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the shared product-state design-system migration by adapting the dedicated WorldBookEmptyState wrapper to render the canonical components/ui/feedback/EmptyState primitive. Preserve the current World Books first-run content, 3-step flow, example text, template quick-start actions, import action, callbacks, and translation fallbacks. Keep scope limited to WorldBookEmptyState, its focused tests, and the product-state baseline entry for this component.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WorldBookEmptyState renders the canonical EmptyState design-system marker while preserving title, description, steps, example copy, template buttons, import action, callbacks, and i18n fallbacks.
- [x] #2 Focused tests cover the canonical EmptyState marker plus the existing WorldBookEmptyState behavior for steps, example copy, template quick starts, primary create action, and import action.
- [x] #3 The product-state guard passes without the WorldBookEmptyState local-empty-state baseline debt, and only the migrated stale baseline entry is removed.
- [x] #4 Scope remains limited to the WorldBookEmptyState wrapper, its focused tests, and the design-system baseline; no unrelated World Books manager or broader empty-state migrations are included.
- [x] #5 Focused Vitest coverage, design-system state verification, git diff checks, and Bandit applicability are recorded before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add/adjust the focused WorldBookEmptyState test first so it requires the canonical EmptyState marker while preserving existing behavior assertions. 2. Run the focused test to verify the new assertion fails against the current local wrapper. 3. Adapt WorldBookEmptyState to compose components/ui/feedback/EmptyState and shared Button-compatible actions while preserving current copy, template callbacks, import callback, layout intent, and translation defaults. 4. Remove the migrated WorldBookEmptyState local-empty-state baseline entry only after the component renders the canonical primitive. 5. Rerun the focused WorldBookEmptyState, FeatureEmptyState, and product-state guard tests plus bun run verify:design-system-state and git diff --check; document frontend-only Bandit skip if applicable.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline before implementation: after installing apps/packages/ui dependencies in the new worktree, bunx vitest run src/components/Option/WorldBooks/__tests__/WorldBookEmptyState.test.tsx src/components/Common/__tests__/FeatureEmptyState.test.tsx src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1 --reporter=dot passed 47/47. bun run verify:design-system-state exited 0 with 523 allowed legacy exceptions and WorldBookEmptyState still present as local-empty-state debt.

TDD red/green: added a canonical EmptyState marker assertion to WorldBookEmptyState.test.tsx, then verified it failed against the local wrapper with received null for data-ds-component=EmptyState. Adapted WorldBookEmptyState to render components/ui/feedback/EmptyState and shared Button actions while preserving current title, description, steps, example copy, template quick starts, create callback, import callback, and translation fallbacks.

Verification after implementation: focused bunx vitest run src/components/Option/WorldBooks/__tests__/WorldBookEmptyState.test.tsx passed 6/6; combined focused run for WorldBookEmptyState, FeatureEmptyState, and product-state-guard passed 47/47; bun run verify:design-system-state exited 0 with baseline exceptions reduced from 523 to 522 and local-empty-state reduced from 3 to 2; git diff --check exited 0 before task-finalization edits. Broader bun run test:worldbooks passed 62 test files, 192 tests passed, 6 skipped, with existing jsdom CSS parse warnings. Bandit is not applicable to this frontend-only TypeScript/JSON slice.

PR opened: https://github.com/rmusser01/tldw_server/pull/1349
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adapted WorldBookEmptyState to the canonical EmptyState primitive while keeping the World Books first-run copy, step guidance, example text, template buttons, create action, import action, callbacks, and i18n fallbacks intact. Added focused coverage for the canonical design-system marker and removed the migrated WorldBookEmptyState local-empty-state baseline exception.
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
