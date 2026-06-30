---
id: TASK-284
title: Migrate SourceItemsTable degraded filter label to design-system registry
status: Done
assignee: []
created_date: '2026-05-12 03:10'
labels:
  - design-system
  - frontend
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the hardcoded degraded filter label in SourceItemsTable with the design-system state registry fallback while preserving the filter controls and existing item rendering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The degraded filter control in SourceItemsTable displays the design-system degraded state label.
- [x] #2 Focused tests prove the degraded filter label comes from the registry without source-string assertions.
- [x] #3 The matching canonical-state-label baseline exception is removed and the design-system state guard passes.
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused SourceItemsTable test that mocks the design-system registry and verifies the degraded filter button uses the registry-provided label.
2. Replace the hardcoded degraded filter label with `getDesignSystemState("degraded").label`.
3. Remove the matching `canonical-state-label` baseline exception and verify the product-state guard still passes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Red test first: `bunx vitest run src/components/Option/Sources/__tests__/SourceItemsTable.design-system.test.tsx --reporter=dot` failed because the filter button was still named `Degraded` instead of the mocked registry label.
- The component now reads the degraded label from the design-system state registry while leaving filter values, callbacks, and table rendering unchanged.
- Removed the `canonical-state-label:src/components/Option/Sources/SourceItemsTable.tsx:Degraded` baseline entry.
- Bandit skipped: touched implementation is frontend TypeScript/test JSON only, with no Python code path.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated SourceItemsTable's degraded filter label to the design-system state registry by calling `getDesignSystemState("degraded")` and rendering the registry label in the filter control. Added a focused test that mocks the registry and interacts with the button by accessible name, so the coverage proves behavior without asserting against component source strings.

Removed the now-obsolete canonical state label baseline exception. Verification passed with the focused SourceItemsTable test, existing SourceDetailPage coverage, the product-state guard unit suite, the design-system guard CLI, `git diff --check`, and a touched-path TypeScript error filter.
<!-- SECTION:FINAL_SUMMARY:END -->
