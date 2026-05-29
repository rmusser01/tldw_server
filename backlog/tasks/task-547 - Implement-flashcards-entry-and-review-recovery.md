---
id: TASK-547
title: Implement flashcards entry and review recovery
status: In Progress
labels:
- ux
- flashcards
- implementation
- webui
- extension
modified_files:
- apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx
- apps/tldw-frontend/extension/routes/sidepanel-flashcards.tsx
- apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-availability.test.ts
- apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx
- apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx
- apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx
- apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx
- apps/packages/ui/src/public/_locales/en/option.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR 1 from the narrow flashcards UX remediation plan: fix the direct extension /flashcards route, clean remaining Transfer copy, add selected Study deck to Create drawer handoff, keep Re-rate last card visible after rating, and verify Practice again remains absent when there are no cram cards.

References:
- Plan: Docs/superpowers/plans/2026-05-29-flashcards-narrow-ux-remediation-implementation-plan.md from planning commit d3bb1199ef in the source checkout
- Design: Docs/superpowers/specs/2026-05-29-flashcards-narrow-ux-remediation-design.md from planning commits e01a5da940 and 133bb5ec66

Scope is PR 1 only; do not implement PR 2 dashboard/session-history work in this branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 3 completed: updated the Flashcards Import / Export tab label default and summary copy from Transfer summary to Import/export summary while preserving internal TransferActionSummary naming. Added focused assertions covering the desired summary copy and updated manager consistency assertions for visible Import / Export tab copy.

Task 4 completed: added a one-shot Study-selected deck handoff from Review/Study create routing into ManageTab's Create drawer without reusing the direct Manage route filter state. ManageTab now consumes the handoff into local drawer state, clears it on consume/close/manual create, and FlashcardCreateDrawer applies initialDeckId after reset when opening.

Task 4 follow-up completed: fixed the create handoff workspace reset gap by applying createInitialShowWorkspaceDecks on every new openCreateSignal and clearing selectedWorkspaceId for the create handoff path. Added a ManageTab regression covering prior workspace-scoped Manage state followed by a non-workspace Study create handoff.

Task 4 code-quality follow-up completed: fixed stale URL deck fallback in the create handoff so Create uses only the live Study deck state. Added a manager regression for clearing the Study selector after a review URL deck, and asserted ManageTab consumes the handoff callback when processing a new open signal.

Task 5 completed: kept the visible Re-rate last card control available after a rating advances to the next card's question side by rendering the shared undo action outside the answer-only branch, while retaining the existing completion-state undo action and keyboard shortcut behavior. Added focused regression coverage that rates a card, verifies the re-rate action remains visible on the next card, and confirms clicking it restores the reviewed card question and answer for re-rating.

Task 5 test-hardening follow-up completed: changed the rerate regression from an exact 10-second accessible-name assertion to a countdown-tolerant accessible-name regex plus a separate rendered timer assertion, preserving the prior-card question and answer restoration assertion.

Verification:
- RED Task 4: cd apps/packages/ui && bunx vitest run src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx failed on missing createInitialDeckId handoff and missing drawer initialDeckId behavior.
- GREEN Task 4: same focused Vitest command passed, 36 tests.
- RED Task 4 follow-up: cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx failed because the Create drawer still received workspace-77 after a non-workspace create handoff.
- GREEN Task 4 follow-up: cd apps/packages/ui && bunx vitest run src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx passed, 48 tests.
- RED Task 4 code-quality follow-up: cd apps/packages/ui && bunx vitest run src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx failed because Create still passed deck 12 after clearing the live Study deck selection.
- GREEN Task 4 code-quality follow-up: cd apps/packages/ui && bunx vitest run src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx passed, 49 tests.
- Self-review Task 4 code-quality follow-up: git diff --check passed; git diff --stat reviewed.
- Bandit Task 4 code-quality follow-up: skipped because the follow-up touched TypeScript UI code/tests only, no Python touched.
- RED Task 5: cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx failed because Re-rate last card was not accessible after rating advanced to the next card question side.
- GREEN Task 5: cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx passed, 1 test.
- Adjacent Task 5: cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx had ReviewTab.cram-mode.test.tsx pass, and ReviewTab.create-cta.test.tsx fail only on the known pre-existing queue-state badge snapshot mismatch (expected AntD Tag vs rendered design-system Badge).
- Self-review Task 5: git diff --check passed; git diff --stat reviewed.
- Bandit Task 5: skipped because Task 5 touched TypeScript UI code/tests only, no Python touched.
- GREEN Task 5 test-hardening follow-up: cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx passed, 1 test.
- Self-review Task 5 test-hardening follow-up: git diff --check passed; git diff --stat reviewed.
- Bandit Task 5 test-hardening follow-up: skipped because the follow-up touched TypeScript UI test/backlog files only, no Python touched.
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
