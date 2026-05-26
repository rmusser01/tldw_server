---
id: TASK-510
title: Flashcards UX Phase 3B deck dashboard data proof
status: Done
labels:
- ux
- flashcards
- phase-3
- frontend
modified_files:
- Docs/superpowers/plans/2026-05-25-flashcards-ux-fixes-implementation-plan.md
- apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx
- apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx
- apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx
- apps/packages/ui/src/components/Flashcards/components/DeckStudyDashboard.tsx
- apps/packages/ui/src/components/Flashcards/components/index.ts
- apps/packages/ui/src/components/Flashcards/components/__tests__/DeckStudyDashboard.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/ImportExport/ExportPanel.tsx
- apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.analytics-summary.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Phase 3B flashcards UX remediation slice only if current frontend/backend data supports it: prove deck-level dashboard inputs, then add a compact deck-first study dashboard with Review/Cram/Edit/Scheduler/Export actions. If existing data is insufficient, stop and record the API/data follow-up instead of building a partial dashboard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing client/API data supports deck id, deck name, total, new, learning, due, and mature counts without backend changes.
- [x] #2 Study renders a compact deck dashboard when no active review card is shown.
- [x] #3 Dashboard rows expose direct Review, Cram, Edit, Scheduler, and Export actions.
- [x] #4 Direct actions preselect the target deck in their destination workflows.
- [x] #5 Normal Study deck changes clear one-shot handoff state so Scheduler does not reopen a stale dashboard-selected deck.
- [x] #6 Focused component/tab/manager tests cover dashboard sorting, counts, callbacks, analytics coexistence, and handoffs.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-25-flashcards-ux-fixes-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Data proof: `FlashcardAnalyticsSummary.decks` already exposes `deck_id`, `deck_name`, `total`, `new`, `learning`, `due`, and `mature`, so Phase 3B did not need a backend change or per-deck due-count fanout.
- Added `DeckStudyDashboard` as a compact Study launch surface backed by the existing analytics summary, sorting decks with ready work first and preserving Cram for caught-up decks.
- Wired deck dashboard actions through `FlashcardsManager` to Manage, Scheduler, and Export with one-shot deck handoff keys.
- Added stale handoff protection by clearing one-shot deck handoffs when the user changes the Study deck through the normal selector/review path.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Phase 3B deck dashboard data proof and UI. Study now shows deck-level counts and direct Review/Cram/Edit/Scheduler/Export actions when no card is active; direct workflow handoffs preselect the target deck, and normal Study deck changes clear one-shot handoffs to avoid stale Scheduler selection. Verification: focused dashboard/ReviewTab/manager suite passed 39 tests; broader ReviewTab suite passed 57 tests; design-system product-state guard passed; git diff --check passed. Typecheck was attempted with `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` and still fails on existing repo-wide baseline errors outside this slice. Bandit skipped because this slice touched frontend TypeScript/TSX tests, docs, and Backlog files only.
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
