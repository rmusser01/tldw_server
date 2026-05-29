---
id: TASK-545
title: Migrate Flashcards document row product states to design system
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-29 03:46'
labels:
  - flashcards
  - webui
  - design-system
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the remaining Flashcards document row Alert and Tag product-state affordances from Ant Design primitives to shared design-system primitives without broadening beyond the /flashcards document row workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FlashcardDocumentRow no longer imports or renders Ant Design Alert or Tag for product-state row affordances.
- [x] #2 Upload errors, conflict recovery, validation/not-found errors, saving status, card-type chips, deck chips, tag chips, and source chips preserve visible copy, actions, and test ids where present.
- [x] #3 Focused Flashcards document-row tests cover the design-system Alert and chip rendering paths.
- [x] #4 The design-system product-state baseline no longer lists FlashcardDocumentRow Alert or Tag findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect existing FlashcardDocumentRow tests and local design-system primitives used for chips/badges.
2. Add focused failing regression coverage for row alert and chip product-state rendering.
3. Replace Ant Design Alert/Tag usages in FlashcardDocumentRow with shared design-system primitives while preserving behavior.
4. Remove resolved FlashcardDocumentRow entries from the product-state baseline and run focused verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced FlashcardDocumentRow Ant Design Alert usage with shared design-system Alert for upload errors, conflict recovery, and validation/not-found errors.
- Replaced FlashcardDocumentRow direct Ant Design Tag usage with shared design-system Badge for saving status, card type, deck, user tags, and source chips.
- Migrated FlashcardQueueStateBadge from Ant Design Tag to shared design-system Badge so row queue-state metadata uses the same design-system chip contract.
- Added focused row tests for design-system metadata badges, conflict alerts, and upload-error alerts.
- Removed the resolved FlashcardDocumentRow Alert/Tag exceptions from the design-system product-state baseline.

Verification:
- RED: bun run test src/components/Flashcards/components/__tests__/FlashcardDocumentRow.test.tsx src/components/Flashcards/components/__tests__/FlashcardDocumentRow.image-insert.test.tsx --maxWorkers=1 --no-file-parallelism failed on the new design-system Alert/Badge assertions.
- GREEN: bun run test src/components/Flashcards/components/__tests__/FlashcardDocumentRow.test.tsx src/components/Flashcards/components/__tests__/FlashcardDocumentRow.image-insert.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.document-mode.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.queue-state.test.tsx --maxWorkers=1 --no-file-parallelism passed 15 tests across 4 files.
- git diff --check passed.
- rg confirmed no FlashcardDocumentRow entries remain in design-system-product-state-baseline.json.
- Bandit skipped: no Python files touched.
- bun run verify:design-system-state still exits 1 on unrelated existing non-Flashcards blocked/stale findings; Flashcards exceptions dropped from 28 to 24.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Flashcards document row product-state alerts and chips now render through shared design-system Alert/Badge primitives, including queue-state badges. Focused row, document-mode, and review queue-state tests cover the migrated paths. The FlashcardDocumentRow Alert/Tag baseline exceptions were removed; repo-wide design-system verification still reports unrelated existing non-Flashcards blocked/stale findings.
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
