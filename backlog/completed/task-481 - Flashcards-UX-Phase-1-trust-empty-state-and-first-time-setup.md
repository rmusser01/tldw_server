---
id: TASK-481
title: Flashcards UX Phase 1 trust empty state and first-time setup
status: Done
labels:
- ux
- flashcards
- frontend
- tests
modified_files:
- apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx
- apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx
- apps/packages/ui/src/components/Flashcards/components/StudyPackCreateDrawer.tsx
- apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ManageTab.first-time.test.tsx
- apps/packages/ui/src/components/Flashcards/components/__tests__/StudyPackCreateDrawer.test.tsx
- apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts
- apps/tldw-frontend/e2e/utils/page-objects/FlashcardsPage.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Phase 1 from Docs/superpowers/plans/2026-05-25-flashcards-ux-fixes-implementation-plan.md in the isolated flashcards worktree. Scope: make first entry to /flashcards understandable before dense transfer, management, or scheduler tooling. Keep changes scoped to /flashcards first-time setup, Manage no-card state, transfer labeling, scheduler discoverability, and Study Pack source-ID copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Empty accounts land on Study/home, not dense transfer tooling, while explicit generate/study-pack deep links still open transfer setup.
- [x] #2 First screen explains flashcards in this product and exposes clear actions to create manually, import a deck, generate from source, or choose an existing deck when present.
- [x] #3 Manage no-card state suppresses expert filters, sort, density, and shortcut chips until cards exist or filters are active.
- [x] #4 Scheduler is discoverable before a deck exists through disabled-tab copy or Study empty-state scheduling preview.
- [x] #5 Transfer/create/import label no longer implies normal import/export is LLM-only.
- [x] #6 Study Pack setup copy labels Source ID as an advanced/manual source reference and points users toward supported source types.
- [x] #7 Focused component/e2e verification is run or limitations are documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED component tests for Phase 1 first-entry tab behavior, first-time Study guidance/actions, Manage no-card chrome suppression, transfer label copy, scheduler discoverability, and Study Pack Source ID copy.
2. Implement the smallest scoped UI changes in FlashcardsManager, ReviewTab, ManageTab, ImportExportTab/StudyPackPanel needed to satisfy the tests.
3. Add or adjust Playwright coverage for first-time/empty/tab behavior using the Phase 0 harness selectors.
4. Run focused Vitest and Playwright verification, record results, and complete the Backlog task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 1 complete. Empty /flashcards accounts now stay on Study with first-time setup actions and Scheduler preview copy; explicit generate and valid Study Pack deep links still open Transfer. Manage suppresses expert chrome for a no-card first-run state, Transfer no longer carries a visible LLM badge, and Study Pack Source ID copy is labeled as advanced/manual. Verification: Phase 1 RED tests observed failing before implementation; targeted affected component files passed; full Flashcards component suite passed (69 files, 333 tests); focused Phase 1 Playwright smoke passed after escalation for local Next port binding. Bandit skipped: TS/TSX/tests/backlog only.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched Python code or non-Python skip documented
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
