---
id: TASK-45.44.9.8
title: Migrate Flashcards ExportPanel preview alert to design-system Alert
status: Done
labels:
- design-system
- webui
- extension
- product-state
- flashcards
priority: medium
parent_task_id: TASK-45.44.9
references:
- https://github.com/rmusser01/tldw_server/issues/1666
- apps/packages/ui/src/components/Flashcards/tabs/ImportExport/ExportPanel.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
modified_files:
- apps/packages/ui/src/components/Flashcards/tabs/ImportExport/ExportPanel.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the Flashcards Import/Export export-preview callout off AntD Alert and onto the canonical design-system Alert while preserving preview copy and export behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ExportPanel preview callout renders the design-system Alert primitive instead of AntD Alert.
- [x] #2 Focused ImportExportTab export coverage proves the preview copy remains visible and wrapped in the canonical design-system marker.
- [x] #3 Design-system product-state verifier passes with the stale ExportPanel Alert baseline entry removed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing focused ImportExportTab export-preview assertion requiring the preview callout to render with the design-system Alert marker.
2. Replace the ExportPanel AntD Alert preview usage with the canonical design-system Alert primitive while preserving title, description, data-testid, and export behavior.
3. Remove the ExportPanel Alert entry from the product-state baseline and run focused tests plus the design-system verifier.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD red/green completed. Added a focused ImportExportTab export-preview assertion requiring the existing preview callout to carry the canonical data-ds-component Alert marker; the red run failed because the AntD Alert preview lacked that marker. Replaced only the export preview callout with the shared design-system Alert primitive while preserving title, preview description interpolation, data-testid, and export behavior, then removed the stale ExportPanel Alert baseline entry.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the Flashcards Import/Export export preview callout from AntD Alert to the design-system Alert primitive. Focused export coverage now verifies the preview copy remains visible and the preview root carries data-ds-component="Alert", and the product-state baseline no longer contains the ExportPanel Alert exception. Verification: red ImportExportTab import-results test failed on the missing design-system marker; green ImportExportTab import-results suite passed 22/22; product-state guard passed 54/54; bun run verify:design-system-state passed with 263 allowed legacy exceptions and 39 remaining Flashcards/Quiz/study-flow exceptions; baseline JSON parse reported targetRows 0; git diff --check passed. TypeScript still exits 2 on 330 existing diagnostics, with no diagnostics for ExportPanel, ImportExportTab.import-results, the baseline, or TASK-45.44.9.8. Bandit skipped because this slice touched frontend TSX/test/JSON/Backlog markdown only.
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
