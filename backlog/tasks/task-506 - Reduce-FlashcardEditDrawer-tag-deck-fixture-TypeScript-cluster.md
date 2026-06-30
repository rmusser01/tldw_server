---
id: TASK-506
title: Reduce FlashcardEditDrawer tag deck fixture TypeScript cluster
status: Done
references:
- TASK-505
- apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.tags.test.tsx
- apps/packages/ui/src/services/flashcards.ts
- apps/packages/ui/tsconfig.json
modified_files:
- apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.tags.test.tsx
- backlog/tasks/task-506 - Reduce-FlashcardEditDrawer-tag-deck-fixture-TypeScript-cluster.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained FlashcardEditDrawer tags test deck fixture cluster. Current package `tsc` output reports three errors in `src/components/Flashcards/components/__tests__/FlashcardEditDrawer.tags.test.tsx` because the local `Deck[]` fixture lacks the required `review_prompt_side` field.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current FlashcardEditDrawer tags compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to test fixture drift rather than production behavior.
- [x] #3 The `FlashcardEditDrawer.tags.test.tsx` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused behavior test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red compiler evidence came from `/tmp/task505-tsc-final.txt`, which contained three `FlashcardEditDrawer.tags.test.tsx` diagnostics because the local deck fixture was missing the required `review_prompt_side` field.
- Root cause was test fixture drift after `Deck.review_prompt_side` became required. Production flashcard behavior was not changed.
- Added `review_prompt_side: "front"` to the local `decks` fixture, matching the default used by nearby flashcard tests.
- Focused test: `bunx vitest run src/components/Flashcards/components/__tests__/FlashcardEditDrawer.tags.test.tsx` from `apps/packages/ui` passed 3/3.
- Package compiler capture: `bunx tsc --noEmit --pretty false > /tmp/task506-tsc-final.txt 2>&1` from `apps/packages/ui` still exits 2 for the known baseline, but `error TS` lines reduced from 93 to 90 and `rg -n 'FlashcardEditDrawer\.tags' /tmp/task506-tsc-final.txt` returned no matches.
- Bandit skipped: this is a TypeScript test-only change and Bandit is a Python security scanner; no Python touched scope exists for this task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the three-error `FlashcardEditDrawer.tags.test.tsx` package `tsc` cluster by adding the required deck review prompt side to the local fixture. The shared UI baseline is now 90 `error TS` lines after this slice.
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
