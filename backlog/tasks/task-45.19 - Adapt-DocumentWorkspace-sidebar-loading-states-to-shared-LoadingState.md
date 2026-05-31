---
id: TASK-45.19
title: Adapt DocumentWorkspace sidebar loading states to shared LoadingState
status: Done
assignee: []
created_date: '2026-05-09 00:06'
updated_date: '2026-05-09 00:43'
labels:
  - design-system
  - webui
  - guard
  - document-workspace
dependencies: []
references:
  - >-
    apps/packages/ui/src/components/DocumentWorkspace/LeftSidebar/QuickInsightsTab.tsx
  - >-
    apps/packages/ui/src/components/DocumentWorkspace/LeftSidebar/ReferencesTab.tsx
  - apps/packages/ui/src/components/ui/feedback/LoadingState.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finish the current local-loading-state migration queue by routing the DocumentWorkspace left-sidebar QuickInsights and References loading adapters through the shared LoadingState primitive while preserving their loading branch behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 QuickInsightsTab loading branch renders through shared LoadingState.
- [x] #2 ReferencesTab loading branch renders through shared LoadingState.
- [x] #3 The product-state guard baseline no longer contains DocumentWorkspace local-loading-state exceptions.
- [x] #4 Focused sidebar tests and the design-system product-state verifier pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused loading-branch coverage for QuickInsightsTab and ReferencesTab.
2. Replace the local AntD Skeleton wrappers with the shared design-system LoadingState primitive while preserving sidebar spacing and approximate skeleton density.
3. Remove the migrated local-loading-state baseline exceptions and refresh only the adjacent ReferencesTab AntD Empty baseline id affected by line movement.
4. Verify with focused Vitest, product-state guard tests, product-state verifier, and diff whitespace checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD red path: focused sidebar tests failed before implementation because neither loading branch rendered data-ds-component="LoadingState". Implementation replaced the local Skeleton wrappers with SharedLoadingState in both sidebar tabs.

Verification: focused sidebar Vitest passed (2 files, 4 tests); product-state guard Vitest passed (42 tests); bun run verify:design-system-state passed with 516 allowed legacy exceptions and no local-loading-state bucket; git diff --check passed. Bandit was not run because the touched implementation scope is TypeScript/TSX, JSON, and Backlog metadata with no Python files.

PR review follow-up: Gemini flagged that a single shared LoadingState collapsed the original per-item loading structure in QuickInsightsTab and ReferencesTab. CodeRabbit flagged hardcoded store teardown in the QuickInsights loading-state test. Reopening this task to address those comments on the same PR branch.

Review fix verification: focused sidebar Vitest passed (2 files, 4 tests); product-state guard Vitest passed (42 tests); bun run verify:design-system-state passed after refreshing the ReferencesTab Empty baseline id; git diff --check passed. Bandit remains not applicable because no Python files are touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adapted the DocumentWorkspace left-sidebar QuickInsights and References loading branches to render through the shared design-system LoadingState primitive instead of local AntD Skeleton wrappers. The PR review follow-up now preserves the original per-item loading structure by rendering three shared placeholders for QuickInsights and four shared placeholders for References, left-aligned within the existing spaced containers.

Added focused tests for both loading branches, strengthened those tests to assert the expected shared LoadingState counts, and updated the QuickInsights loading test to snapshot and restore the exact Zustand store states between tests. Removed the migrated local-loading-state baseline entries and refreshed ReferencesTab AntD Empty baseline ids caused by line movement.

Verification: bunx vitest run src/components/DocumentWorkspace/LeftSidebar/__tests__/QuickInsightsTab.loading-state.test.tsx src/components/DocumentWorkspace/LeftSidebar/__tests__/ReferencesTab.test.tsx --reporter=dot; bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot; bun run verify:design-system-state; git diff --check. Bandit was not applicable because no Python files were touched.
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
