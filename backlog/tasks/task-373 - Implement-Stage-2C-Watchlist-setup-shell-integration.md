---
id: TASK-373
title: Implement Stage 2C Watchlist setup shell integration
status: Done
assignee: []
created_date: '2026-05-15 04:56'
updated_date: '2026-05-15 05:44'
labels:
  - watchlists
  - stage2
  - frontend
dependencies:
  - TASK-372
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage2-setup-wizard-plan.md
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage2-setup-wizard-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate the Stage 2 setup wizard into the Watchlists shell create flow. Scope: replace create-mode modal with setup wizard, keep edit modal for metadata edits, use existing service functions to create Watchlist/source/job with watchlist_id propagation, select the created Watchlist, and route to the correct scoped tab. No Overview quick setup repositioning in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Primary create control opens the Stage 2 setup wizard while edit keeps metadata modal behavior.
- [x] #2 Topic-only completion creates/selects a Watchlist and routes to Feeds or the planned destination.
- [x] #3 Source-backed/report-goal completion sends watchlist_id on source and job payloads.
- [x] #4 Existing selected-scope route/service tests remain green.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 2C started after Stage 2B commit eece4a843. Scope is shell create-flow integration only: replace create-mode metadata modal with WatchlistSetupWizard, keep edit modal behavior, attach watchlist_id to source/job payloads, update store selection, and route to destination tab.

Stage 2C TDD result: red shell tests failed as expected because the create button still opened the old metadata modal and the wizard preset buttons were absent. Implemented shell integration by opening WatchlistSetupWizard from the stable watchlists-create-container control, leaving edit on the existing metadata modal, adapting createWatchlist/createWatchlistSource/bulkCreateSources/createWatchlistJob, attaching watchlist_id to source/job payloads, adding/selecting the created Watchlist, and routing to the wizard destination tab. Added wizard submit error feedback for service failures. Green verification: ./node_modules/.bin/vitest run src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx src/components/Option/Watchlists/__tests__/watchlists-selected-scope-contract.test.ts --maxWorkers=1 --no-file-parallelism --reporter=verbose -> 2 files passed, 11 tests passed. Wizard/model/copy regression also passed: 3 files, 12 tests. git diff --check passed. Bandit not applicable because this task touched only frontend TypeScript/TSX and Backlog task files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Integrated the Stage 2 Watchlist setup wizard into the Watchlists shell create flow. The primary create control now opens the Watchlist-first wizard while edit continues using the metadata modal. Topic-only completion creates/selects the Watchlist and routes to Feeds. Source-backed setup attaches watchlist_id to source payloads via createWatchlistSource or bulkCreateSources and attaches watchlist_id to createWatchlistJob before routing to Monitors. Completion adds the new Watchlist to the store, selects it, and shows the existing success notification. Verification: the shell integration plus selected-scope contract tests passed (2 files, 11 tests), the wizard/model/copy regression suite passed (3 files, 12 tests), and git diff --check passed. Bandit was skipped as not applicable for frontend-only TS/TSX changes. Overview quick setup repositioning remains intentionally deferred to TASK-374.
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
