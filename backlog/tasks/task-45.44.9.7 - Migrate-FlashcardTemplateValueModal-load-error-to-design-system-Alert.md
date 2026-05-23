---
id: TASK-45.44.9.7
title: Migrate FlashcardTemplateValueModal load error to design-system Alert
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
- apps/packages/ui/src/components/Flashcards/components/FlashcardTemplateValueModal.tsx
- apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.templates.test.tsx
- apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardTemplateValueModal.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
modified_files:
- apps/packages/ui/src/components/Flashcards/components/FlashcardTemplateValueModal.tsx
- apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardTemplateValueModal.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the FlashcardTemplateValueModal template-load error callout off AntD Alert and onto the canonical design-system Alert while preserving modal behavior and copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FlashcardTemplateValueModal template-load error callout renders the design-system Alert primitive instead of AntD Alert.
- [x] #2 Focused Flashcards coverage proves the error text remains visible and wrapped in the canonical design-system marker.
- [x] #3 Design-system product-state verifier passes with the stale FlashcardTemplateValueModal Alert baseline entry removed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing focused FlashcardCreateDrawer template-modal test assertion requiring the load-error callout to render with the design-system Alert marker.
2. Replace the FlashcardTemplateValueModal AntD Alert usage with the canonical design-system Alert primitive while preserving error title and description copy.
3. Remove the FlashcardTemplateValueModal Alert entry from the product-state baseline and run focused tests plus the design-system verifier.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD red/green completed. Added a focused FlashcardTemplateValueModal load-error test requiring the error title to be wrapped by the canonical data-ds-component Alert marker; the first focused run failed because the existing AntD Alert had no design-system marker. Replaced only the modal template-load error callout with the shared design-system Alert primitive while preserving title and error-message copy, then removed the stale FlashcardTemplateValueModal Alert baseline entry.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the FlashcardTemplateValueModal template-load error callout from AntD Alert to the design-system Alert primitive. Focused coverage now verifies the load-error text renders in the canonical Alert wrapper, and the product-state baseline no longer contains the FlashcardTemplateValueModal Alert exception. Verification: red focused modal test failed on the missing design-system marker; green focused modal test passed 1/1; existing FlashcardCreateDrawer template-flow suite passed 8/8; product-state guard passed 54/54; bun run verify:design-system-state passed with 264 allowed legacy exceptions and 40 remaining Flashcards/Quiz/study-flow exceptions; baseline JSON parse reported targetRows 0; git diff --check passed. TypeScript still exits 2 on 330 existing diagnostics, with no diagnostics for FlashcardTemplateValueModal, the new modal test, the baseline, or TASK-45.44.9.7. Bandit skipped because this slice touched frontend TSX/test/JSON/Backlog markdown only.
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
