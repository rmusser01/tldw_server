---
id: TASK-45.44.9.2
title: Migrate FlashcardDocumentView truncation banner to design-system Alert
status: Done
labels:
- design-system
- webui
- extension
- product-state
priority: medium
parent_task_id: TASK-45.44.9
references:
- https://github.com/rmusser01/tldw_server/issues/1666
- apps/packages/ui/src/components/Flashcards/components/FlashcardDocumentView.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/pull/1930
modified_files:
- apps/packages/ui/src/components/Flashcards/components/FlashcardDocumentView.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ManageTab.document-mode.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the FlashcardDocumentView truncated-results warning banner off AntD Alert and onto the canonical design-system Alert, with focused coverage and verifier evidence for the baseline reduction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Replace the AntD Alert import and JSX in FlashcardDocumentView with the canonical design-system Alert primitive.
2. Extend the existing document-mode truncation test to assert the rendered design-system Alert marker and preserved copy.
3. Remove the stale product-state baseline entry and run the focused verifier/test commands.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed a narrow Flashcards product-state migration slice and opened PR #1930. Verification: `bun run verify:design-system-state` reports 321 baseline exceptions with no stale FlashcardDocumentView entry; `bunx vitest run src/components/Flashcards/tabs/__tests__/ManageTab.document-mode.test.tsx` passes 3 tests; `git diff --check` passes. Bandit skipped because this slice only changes frontend TSX/test/baseline JSON.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
