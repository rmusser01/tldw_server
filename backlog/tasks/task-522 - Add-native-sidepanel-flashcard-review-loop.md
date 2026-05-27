---
id: TASK-522
title: Add native sidepanel flashcard review loop
status: Done
labels:
- flashcards
- extension
- ux
priority: medium
references:
- Flashcards-UX-Fix-List.md
- Docs/superpowers/plans/2026-05-25-flashcards-ux-fixes-implementation-plan.md
documentation:
- Docs/superpowers/plans/2026-05-27-flashcards-extension-native-review-implementation-plan.md
modified_files:
- apps/packages/ui/src/routes/sidepanel-flashcards.tsx
- apps/packages/ui/src/routes/__tests__/sidepanel-flashcards.test.tsx
- apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts
- Flashcards-UX-Fix-List.md
- apps/extension/docs/features/flashcards.md
- Docs/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md
- Docs/Published/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a compact native flashcard review loop to the extension sidepanel so users can review due cards without opening the full Flashcards workspace.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sidepanel exposes a compact review action for due flashcards without requiring full Flashcards navigation.
- [x] #2 Review action reuses existing next-review query and review mutation contracts.
- [x] #3 Users can select a deck, reveal the answer, and submit Again/Hard/Good/Easy ratings.
- [x] #4 Successful rating advances/refetches the next due card and shows session progress/saved state.
- [x] #5 Failed rating submission keeps the current card visible with inline recovery copy.
- [x] #6 Full Flashcards remains the path for imports, management, analytics, and richer review controls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-flashcards-extension-native-review-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation added a compact sidepanel review panel instead of duplicating full Flashcards Study. Full Flashcards remains the richer path for cram, assistant support, analytics, undo/re-rate, imports, and deck management.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented a compact native sidepanel review loop for F12. The sidepanel now exposes Review due card, reuses the existing next-review query and review mutation, lets users select a deck, reveal a due card, submit Again/Hard/Good/Easy ratings, and see inline session progress. Rating failures keep the current card and answer visible with recovery copy, rapid duplicate rating clicks are guarded synchronously, successful ratings hide the already-rated card while invalidation-driven next-card loading advances, and next-card loading failures show distinct retry UI without allowing duplicate re-rating. Updated the master UX fix list plus extension/user docs to remove the deferred in-sidepanel review claim and keep richer review controls scoped to full Flashcards.

PR review fixes:
- Added `refetchOnWindowFocus` support for review queries and disabled it in the sidepanel review panel.
- Switched sidepanel review loading display to initial `isLoading` only so background refetches do not blank the active card.
- Removed the normal explicit post-submit `reviewQuery.refetch()` path to avoid redundant next-card fetches; retry still uses `refetch` after an advance failure.
- Logged review submission and retry-load errors with safe card/rating context.
- Added all-four rating mapping tests, next-card-load failure tests, and backlog/doc copy fixes requested on PR #2083.

Verification:
- RED: `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx` failed on the missing Review due card action after dependency install.
- RED duplicate guard: `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx -t "ignores duplicate sidepanel review ratings"` failed with two mutation calls before the synchronous guard.
- GREEN focused after review fixes: `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx` passed 44 tests.
- Related regression after review fixes: `bunx vitest run src/components/Flashcards/hooks/__tests__/useFlashcardQueries.review-next.test.tsx src/routes/__tests__/sidepanel-flashcards.test.tsx src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts` passed 54 tests.
- Typecheck: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still fails on unrelated baseline `CharacterListContent.design-system.test.tsx(35,3)` (`"comfortable"` not assignable to `GalleryCardDensity`).
- Diff hygiene: `git diff --check` passed.
- Bandit: not applicable; no Python files touched.
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
