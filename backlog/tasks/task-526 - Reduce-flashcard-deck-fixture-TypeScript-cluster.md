---
id: TASK-526
title: Reduce flashcard deck fixture TypeScript cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:57'
labels: []
dependencies: []
references:
  - TASK-525
  - >-
    apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardDocumentRow.test.tsx
  - >-
    apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardDocumentRow.image-insert.test.tsx
  - >-
    apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.image-insert.test.tsx
  - >-
    apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.reset-scheduling.test.tsx
  - apps/packages/ui/src/services/flashcards.ts
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the four test-only diagnostics in flashcard component tests. Current package `tsc` output reports `Deck` fixtures missing the required `review_prompt_side` field.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current flashcard deck fixture compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to stale test `Deck` fixtures rather than behavior changes.
- [x] #3 The four flashcard deck fixture diagnostics are removed from package `tsc` output.
- [x] #4 Focused flashcard component tests are run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task525-tsc-final.txt`: package `tsc` reported four diagnostics where flashcard component test `Deck` fixtures were missing the required `review_prompt_side` field.
- Root cause was stale test fixture data only. The `Deck` type now requires `review_prompt_side`, and nearby flashcard tests already use the default `"front"` value.
- Added `review_prompt_side: "front"` to the four affected deck fixtures without changing component behavior or assertions.
- Focused verification: `bunx vitest run src/components/Flashcards/components/__tests__/FlashcardDocumentRow.test.tsx src/components/Flashcards/components/__tests__/FlashcardDocumentRow.image-insert.test.tsx src/components/Flashcards/components/__tests__/FlashcardEditDrawer.image-insert.test.tsx src/components/Flashcards/components/__tests__/FlashcardEditDrawer.reset-scheduling.test.tsx` passed: 6 tests.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task526-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 46 in `/tmp/task525-tsc-final.txt` to 42 in `/tmp/task526-tsc-final.txt`; searching for the four flashcard paths and `review_prompt_side` returns no matches.
- Bandit skipped: this is a TypeScript test-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the four flashcard component test `Deck` fixture diagnostics by adding the required `review_prompt_side: "front"` field to the stale fixtures. Focused Vitest passed with 6 tests, and package `tsc` baseline dropped from 46 to 42 with no remaining flashcard deck fixture diagnostics.
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
