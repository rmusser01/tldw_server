---
id: TASK-45.14
title: Adapt SearchEmptyState to shared EmptyState
status: Done
assignee: []
created_date: '2026-05-07 01:51'
updated_date: '2026-05-07 01:56'
labels:
  - design-system
  - frontend
  - knowledge
dependencies: []
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
  - Docs/Design/tldw_web_design_system_inventory.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the shared product-state design-system migration by adapting the Knowledge SearchEmptyState wrapper to render the canonical components/ui/feedback/EmptyState primitive. Preserve the four existing variants (initial, no-results, timeout, disconnected), copy, retry callback, dismiss hint callback, showHint behavior, and i18n fallbacks across SearchTab, FileSearchTab, and QASearchTab consumers. Keep scope limited to SearchEmptyState, focused tests, and the product-state baseline entry for this component.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SearchEmptyState renders the canonical EmptyState design-system marker for initial, no-results, timeout, and disconnected variants while preserving variant-specific copy and icon intent.
- [x] #2 Focused tests cover dismiss hint behavior for initial, retry behavior for timeout, absent retry action when no callback is provided, no-results copy, disconnected copy, and the canonical EmptyState marker.
- [x] #3 The product-state guard passes without the SearchEmptyState local-empty-state baseline debt, and only the migrated stale baseline entry is removed.
- [x] #4 Scope remains limited to SearchEmptyState, its direct focused test, and the design-system baseline; no unrelated Knowledge search behavior or consumer rewrites are included.
- [x] #5 Focused Vitest coverage, design-system state verification, git diff checks, and Bandit applicability are recorded before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused SearchEmptyState test that requires the canonical EmptyState marker while preserving current variant copy and action behavior. 2. Run the focused test to verify the canonical-marker assertion fails against the current local wrapper. 3. Adapt SearchEmptyState to compose components/ui/feedback/EmptyState and shared Button-compatible actions while preserving current variants, callbacks, showHint behavior, and translation defaults. 4. Remove the migrated SearchEmptyState local-empty-state baseline entry only after the component renders the canonical primitive. 5. Rerun focused SearchEmptyState tests, nearby Knowledge tests, product-state guard tests, bun run verify:design-system-state, and git diff --check; document frontend-only Bandit skip if applicable.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline before implementation: after installing apps/packages/ui dependencies in the new worktree, bunx vitest run src/components/Knowledge/__tests__/KnowledgeTabs.test.tsx src/components/Knowledge/__tests__/KnowledgePanelTabRouting.test.tsx src/components/Knowledge/QASearchTab/__tests__/GeneratedAnswerCard.test.tsx src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1 --reporter=dot passed 45/45 with existing tldw server not configured request warnings. bun run verify:design-system-state exited 0 with 522 allowed legacy exceptions and SearchEmptyState still present as local-empty-state debt.

TDD red/green: added a focused SearchEmptyState test covering the canonical EmptyState marker for initial, no-results, timeout, and disconnected variants plus dismiss/retry action behavior. The red run failed as expected because data-ds-component=EmptyState was missing from the current local wrapper for all variants. Adapted SearchEmptyState to render components/ui/feedback/EmptyState and removed direct AntD Button usage while preserving variant copy, retry callback, dismiss callback, showHint behavior, and translation fallbacks.

Verification after implementation: focused SearchEmptyState test passed 5/5; combined focused run for SearchEmptyState, nearby Knowledge tests, GeneratedAnswerCard, and product-state-guard passed 50/50 with existing tldw server not configured request warnings; bun run verify:design-system-state exited 0 with baseline exceptions reduced from 522 to 521 and local-empty-state reduced from 2 to 1; broader bunx vitest run src/components/Knowledge --maxWorkers=1 --reporter=dot passed 12 test files and 38 tests with existing tldw server not configured request warnings; git diff --check exited 0 before final task-record edits. Bandit is not applicable to this frontend-only TypeScript/JSON slice.

PR opened: https://github.com/rmusser01/tldw_server/pull/1350
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adapted SearchEmptyState to the canonical EmptyState primitive while preserving the initial, no-results, timeout, and disconnected variants, including copy, retry behavior, dismiss hint behavior, showHint gating, and i18n fallbacks. Added focused coverage for the canonical design-system marker and removed the migrated SearchEmptyState local-empty-state baseline exception.
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
