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

Verification:
- RED Task 4: cd apps/packages/ui && bunx vitest run src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx failed on missing createInitialDeckId handoff and missing drawer initialDeckId behavior.
- GREEN Task 4: same focused Vitest command passed, 36 tests.
- Self-review Task 4: git diff --check passed; git diff --stat reviewed.
- Bandit Task 4: skipped because Task 4 touched TypeScript UI code/tests only, no Python touched.
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
