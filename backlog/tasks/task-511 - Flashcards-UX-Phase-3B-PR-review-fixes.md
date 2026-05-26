---
id: TASK-511
title: Flashcards UX Phase 3B PR review fixes
status: Done
labels:
- ux
- flashcards
- phase-3
- frontend
- review-fix
modified_files:
- backlog/tasks/task-511 - Flashcards-UX-Phase-3B-PR-review-fixes.md
- apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx
- apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx
- apps/packages/ui/src/components/Flashcards/components/DeckStudyDashboard.tsx
- apps/packages/ui/src/components/Flashcards/components/__tests__/DeckStudyDashboard.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address PR #2068 review feedback for the Phase 3B deck study dashboard: harden dashboard row data handling, clear stale one-shot deck handoffs across all deck-change paths, and prevent dashboard analytics from firing during review/cram loading gaps.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dashboard row building tolerates missing deck arrays and missing deck names without crashing.
- [x] #2 Any deck-change path clears stale one-shot handoffs before applying a new Manage/Scheduler/Export handoff.
- [x] #3 Scheduler no longer prefers an old dashboard handoff after Manage, Export, Manage-card-review, route, or Study selector deck changes.
- [x] #4 Dashboard analytics query is disabled while review or cram data is still loading.
- [x] #5 Focused regression tests cover all PR review comments.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Addressed Gemini's defensive data comment by allowing `DeckStudyDashboard` to receive null/undefined deck lists and falling back to `Deck {id}` when analytics rows do not include a usable deck name.
- Addressed Qodo's stale handoff bug by routing selector, dashboard action, Manage review-card, and URL-driven deck changes through a shared deck-change helper that clears one-shot handoffs before setting the next deck.
- Addressed Qodo's dashboard analytics loading-gap bug by disabling the global dashboard analytics query while the active review-card or cram queue query is still loading/fetching.
- Added red-first regression coverage for all three review items before production fixes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2068 review fixes are implemented. The dashboard row builder is defensive against incomplete analytics/deck data, all deck-change paths now clear stale one-shot handoffs before applying a new destination handoff, and dashboard analytics no longer fetches during transient review/cram loading gaps. Verification: focused review-fix suite passed 42 tests after red failures; broader ReviewTab suite passed 58 tests; design-system product-state guard passed; `git diff --check` passed. Bandit skipped because this slice only touched frontend TypeScript/TSX tests and Backlog files.
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
