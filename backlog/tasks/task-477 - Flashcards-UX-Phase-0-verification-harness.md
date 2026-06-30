---
id: TASK-477
title: Flashcards UX Phase 0 verification harness
status: Done
labels:
- flashcards
- ux
- e2e
- tests
modified_files:
- apps/tldw-frontend/e2e/utils/page-objects/FlashcardsPage.ts
- apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts
- apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Phase 0 of the flashcards UX remediation plan: add stable flashcards page-object selectors, prove or disprove the create-drawer submit issue with backend-backed e2e coverage, add failed-create component coverage, and add keyboard-only review e2e coverage without Phase 2 undo/re-rate scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Add stable page-object selectors needed for create drawer and keyboard review tests.
- [x] Add backend-backed e2e coverage proving create-drawer submit works for an unscoped card.
- [x] Add component coverage proving failed create keeps the drawer open and retains entered fields.
- [x] Add keyboard-only review e2e coverage for Space to show answer and 3 to rate Good.
- [x] Record broader-run blockers separately instead of silently folding them into Phase 0.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-25-flashcards-ux-fixes-implementation-plan.md#phase-0-verification-harness-and-create-drawer-proof
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added create-drawer page-object selectors for the accessible dialog, front/back textareas, create buttons, and message hooks.
- Added review page-object selectors for Good/Easy rating controls and completion/empty states.
- Added `should review a seeded card using only keyboard shortcuts`, which seeds a deck/card through the API, opens Review directly, presses Space, presses 3, and verifies the review POST succeeds.
- Added `creates a flashcard from the drawer and shows it in Manage`, which opens the Manage drawer, submits a front/back card, verifies the create POST, and waits for the new row.
- Added component regression coverage for create failure recovery: error message is shown, the drawer remains open, entered front/back text is retained, success/close callbacks are not fired, and the Create button is re-enabled.
- No production component changes were retained. A temporary Drawer width probe was reverted because it introduced an AntD deprecation warning and did not resolve the drawer select behavior.
- Broader `--grep "flashcard"` e2e run exposed follow-up blockers: review POST can hit backend 429 after repeated local runs, and the existing tag-suggestion drawer path still hits the create-drawer deck-select/off-viewport behavior. These are documented for the next phase rather than treated as resolved here.
- Bandit was not run because the retained changes are frontend TypeScript tests/page objects and a Backlog task record, with no Python runtime code touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 0 harness is implemented for create-submit, create-failure recovery, and keyboard-only review. Verification recorded: component drawer test suite passed 8/8; focused Playwright e2e for keyboard review, drawer create, and shortcuts passed 3/3; `git diff --check` passed. Broader grep was run and documented with unresolved blockers.
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
