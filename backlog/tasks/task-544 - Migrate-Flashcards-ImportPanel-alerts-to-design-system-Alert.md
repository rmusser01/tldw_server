---
id: TASK-544
title: Migrate Flashcards ImportPanel alerts to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-29 01:58'
labels:
  - flashcards
  - webui
  - design-system
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the direct Flashcards import workflow alert states in ImportPanel from Ant Design Alert to the shared design-system Alert primitive without broadening beyond /flashcards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ImportPanel no longer imports or renders Ant Design Alert for flashcards import status states.
- [x] #2 Preflight, structured preview error, and last-result alert states preserve user-facing copy and test ids while rendering the shared design-system Alert marker.
- [x] #3 Focused Flashcards ImportPanel tests cover the design-system alert rendering path.
- [x] #4 The design-system product-state baseline no longer lists ImportPanel Alert findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect existing ImportPanel tests and current alert render paths.
2. Add focused failing regression coverage that verifies ImportPanel alert states render as design-system alerts.
3. Replace the ImportPanel Ant Design Alert usages with the shared Alert primitive, preserving copy/test ids.
4. Remove resolved ImportPanel Alert entries from the product-state baseline and run focused verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced ImportPanel direct Ant Design Alert usage with the shared design-system Alert primitive for preflight warnings, structured preview warnings, and last import results.
- Preserved existing copy, conditional warning/success semantics, help-link behavior, and test ids.
- Added focused import workflow assertions that the preflight warning, structured preview warning, and last-result states render with data-ds-component="Alert".
- Removed the resolved Flashcards ImportPanel Alert exceptions from the design-system product-state baseline.

Verification:
- RED: bun run test src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx --maxWorkers=1 --no-file-parallelism failed on the three new data-ds-component assertions.
- GREEN: bun run test src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx --maxWorkers=1 --no-file-parallelism passed 28 tests.
- git diff --check passed.
- rg confirmed no Flashcards ImportPanel entries remain in design-system-product-state-baseline.json.
- Bandit skipped: no Python files touched.
- bun run verify:design-system-state still exits 1 on unrelated existing non-Flashcards blocked/stale findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Flashcards ImportPanel import status alerts now render through the shared design-system Alert primitive, with focused regression coverage and updated snapshots. The ImportPanel Alert baseline exceptions were removed; repo-wide design-system verification still reports unrelated existing non-Flashcards blocked/stale entries.
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
