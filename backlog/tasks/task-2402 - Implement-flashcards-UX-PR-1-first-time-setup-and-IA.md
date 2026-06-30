---
id: TASK-2402
title: Implement flashcards UX PR 1 first-time setup and IA
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-23 20:47'
labels:
  - ux
  - flashcards
dependencies: []
documentation:
  - >-
    Review fix requested 2026-06-23: remove provider-required copy from Review
    first-time onboarding while preserving Generate/Assistant provider-gating
    surfaces.
  - >-
    Code-quality review fix requested 2026-06-23: make first-run import CTAs
    hand off to the Import file task while generate CTAs intentionally open
    Create/generate
  - preserving the public importExport tab key.
  - >-
    Review fix requested 2026-06-24: normalize Manage search query
    whitespace consistently across cache keys, list requests, document-mode
    queries, and bulk list requests.
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 1 from flashcards remaining UX remediation plan: improve first-time setup, make Scheduler discoverable for zero-deck users, refine Manage empty state ordering, and reorganize Import/Export information architecture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task 1 first-time flashcards UX remediation.

Changed:
- FlashcardsManager keeps Scheduler visible for zero-deck users and renders a Scheduler empty preview for zero-deck Scheduler deep links with create/import actions.
- ManageTab treats query, tags, due status, selected deck, selected workspace, and workspace-deck visibility as active filters; true first-run no-card state shows create/import/generate actions before expert chrome.
- ImportExportTab keeps the `importExport` route key while adding accessible task-first sections for Create and generate plus Import and export, with compact Import/export summary copy.
- Updated/added targeted component tests for zero-deck IA, Manage no-card filter behavior, and Import/Export section decomposition.

Verification:
- Red run after test changes reached expected assertion failures: Scheduler hidden, workspace visibility not active, and missing Import/Export section labels.
- `bunx vitest run src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.empty-state.test.tsx src/components/Flashcards/tabs/__tests__/ImportExportTab.decomposition.test.tsx` passed: 3 files, 42 tests.
- `git diff --check` passed.

Bandit: N/A, frontend-only TypeScript/React changes.

Known skips/blockers:
- Clean worktree dependency symlinks require a temporary ignored `apps/node_modules` link to the main checkout's installed Bun cache for Vitest resolution. The link was removed before staging and is not committed.
- No backend Python touched; no backend tests run.

Review fix follow-up:
- Removed provider-required copy from the Review first-time onboarding guide.
- Added focused ReviewTab coverage asserting the onboarding guide does not render provider-required copy.
- Re-ran provider-gating tests for Generate and Assistant surfaces to confirm those messages remain intact.

Review fix verification:
- Red run: `bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx` failed on the new onboarding assertion because the guide still rendered `LLM provider` copy.
- Green run: `bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.assistant.test.tsx src/components/Flashcards/tabs/__tests__/ImportExportTab.llm-gating.test.tsx` passed: 3 files, 36 tests.
- `git diff --check` passed after the review fix.
- Bandit remains N/A, frontend-only TypeScript/React changes.
- Temporary ignored `apps/node_modules` symlink was used for Vitest resolution and removed before staging.

Code-quality review fix follow-up:
- Added an explicit tokened ImportExport inner-task handoff while keeping the public `importExport` tab key stable.
- Split first-run import and generate callbacks in ReviewTab and ManageTab so import CTAs open Import file and generate CTAs open Create/generate.
- Preserved generate intent, study-pack intent, and deck export handoff behavior with focused ImportExportTab coverage.

Code-quality review fix verification:
- Red run after test changes: `bunx vitest run src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.empty-state.test.tsx src/components/Flashcards/tabs/__tests__/ImportExportTab.decomposition.test.tsx` reached expected failures for empty import task handoffs and shared import/generate callbacks after adding the temporary ignored `apps/node_modules` symlink.
- Green run: same `bunx vitest run ...` command passed: 4 files, 70 tests.
- `git diff --check` passed.
- Bandit remains N/A, frontend-only TypeScript/React changes.
- Temporary ignored `apps/node_modules` symlink was used for Vitest resolution and removed before staging.

Whitespace-query review fix follow-up:
- Rebased PR #2465 onto latest `origin/dev`.
- Added a regression test proving `useManageQuery` trims whitespace for both React Query cache keys and `listFlashcards` request params.
- Normalized Manage search text in `useManageQuery`, document-mode queries, ManageTab filter/query-key inputs, and bulk list requests so whitespace-only search state cannot diverge between UI and backend queries.
- Updated a stale zero-deck Scheduler assertion to match this PR's Scheduler-discoverability behavior after rebasing.

Whitespace-query review fix verification:
- Red run: `bunx vitest run src/components/Flashcards/hooks/__tests__/useFlashcardQueries.tag-suggestions.test.tsx` failed because `listFlashcards` received `q: "  mitochondria  "`.
- Green run: same hook test passed: 1 file, 3 tests.
- Focused PR slice: `bunx vitest run src/components/Flashcards/hooks/__tests__/useFlashcardQueries.tag-suggestions.test.tsx src/components/Flashcards/hooks/__tests__/useFlashcardDocumentQuery.test.ts src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.empty-state.test.tsx src/components/Flashcards/tabs/__tests__/ImportExportTab.decomposition.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx` passed: 6 files, 78 tests.
- `git diff --check` passed.
- Bandit remains N/A, frontend-only TypeScript/React and Backlog metadata changes.
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
