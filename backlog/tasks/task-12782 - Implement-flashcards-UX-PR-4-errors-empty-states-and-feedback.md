---
id: TASK-12782
title: Implement flashcards UX PR 4 errors empty states and feedback
status: Done
labels:
- ux
- flashcards
- frontend
ordinal: 2405
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 4 from the flashcards UX remediation plan: improve error messaging, empty-state hierarchy, and user feedback across flashcards flows before the final responsive/docs phases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Flashcards operations surface clear success, partial failure, retry, and recovery guidance where applicable.
- [x] #2 No-card and filtered-empty states emphasize the correct next action without exposing irrelevant expert chrome.
- [x] #3 Visible feedback distinguishes loading, success, failure, and unavailable states for affected create/import/generate/review flows.
- [x] #4 Focused tests cover the updated states and copy.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation notes:
- Added partial-failure handling for Manage bulk tag updates. Successful cards are no longer presented as a clean success when other selected cards fail; failed cards remain selected and the warning tells the user they can retry.
- Added matching partial-failure handling for Manage bulk move operations while preserving undo for successfully moved cards.
- Code review fix: failed selections produced by select-all-across now remain retryable even when the failed card is off the visible page; the retry path fetches selected failed IDs from the full filtered result set.
- Second code review fix: changing workspace scope now clears the failed-card retry selection so stale off-scope cards are not advertised as retryable.
- Third code review fix: rebased PR #2469 on latest origin/dev, resolved the ManageTab selection reset conflict, preserved retry selections when users toggle visible cards, and changed off-page retry reconstruction to fetch only the missing selected UUIDs instead of refetching all filtered results.
- Labeled the failed-selection recovery state as "failed cards selected for retry" in the selection summary and floating action bar.
- Kept existing first-run and filtered-empty Manage states intact; origin/dev already had the no-card expert-chrome reduction and tests.
- Removed temporary dependency changes after test execution. The worktree still needs the known local `antd` symlink retarget only while running focused Vitest.
- Bandit N/A: frontend-only TypeScript/React changes, no Python touched.

Verification:
- RED: `cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx` failed 2 tests because partial tag/move failures only logged console warnings and did not keep failed cards selected.
- PASS: `cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx` (1 file passed, 10 tests passed).
- PASS: `cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.first-time.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx` (3 files passed, 24 tests passed).
- REVIEW-FIX RED: `cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx` failed the off-page select-all-across retry regression before the selection-scope fix.
- REVIEW-FIX PASS: `cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx` (1 file passed, 11 tests passed).
- REVIEW-FIX PASS: `cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.first-time.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx` (3 files passed, 25 tests passed).
- REVIEW-FIX RED: `cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx` failed the workspace-scope reset regression before workspace visibility and selected-workspace changes were added to the selection reset dependencies.
- REVIEW-FIX PASS: `cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx` (1 file passed, 12 tests passed).
- REVIEW-FIX PASS: `cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.first-time.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx` (3 files passed, 26 tests passed).
- THIRD REVIEW-FIX PASS: `cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx` (1 file passed, 13 tests passed).
- THIRD REVIEW-FIX PASS: `cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ManageTab.undo-stage3.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.first-time.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx` (3 files passed, 27 tests passed).
- THIRD REVIEW-FIX PASS: `git diff --check`.
- THIRD REVIEW-FIX Bandit N/A: frontend-only TypeScript/React and Backlog markdown changes, no Python touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Manage bulk tag and bulk move now report partial failures as warnings with retry guidance, leave failed cards selected for recovery, and avoid clean success messages when only part of the operation completed. Failed selections from select-all-across remain retryable across result pages, visible-card toggles no longer drop off-page retry IDs, retry fetches only the missing selected UUIDs, and workspace scope changes clear stale failed retry selections. Existing first-time and filtered-empty Manage behavior remains covered by focused tests.
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
