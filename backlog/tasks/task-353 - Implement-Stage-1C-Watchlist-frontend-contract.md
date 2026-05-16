---
id: TASK-353
title: Implement Stage 1C Watchlist frontend contract
status: Done
assignee: []
created_date: '2026-05-15 01:56'
updated_date: '2026-05-15 02:02'
labels:
  - watchlists
  - frontend
  - api-client
  - stage1c
dependencies:
  - TASK-352
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage1-implementation-plan.md
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage1-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the frontend contract slice from the Stage 1 first-class Watchlists plan. Scope is limited to shared UI package types, API service methods/query parameters, overview service scoping, and watchlists store state/actions. Do not build the selector shell or child-tab UI in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frontend service tests cover Watchlist CRUD paths and watchlist_id query parameters for existing source/job/run/item/output fetchers.
- [x] #2 watchlists.ts types expose Watchlist domain/status/priority/container/create/update contracts and backward-compatible watchlist_id fields on source/job contracts.
- [x] #3 watchlists.ts service exposes fetch/get/create/update/delete/restore Watchlist methods and scopes child fetchers with watchlist_id when supplied.
- [x] #4 watchlists-overview.ts accepts watchlist_id and forwards it to aggregate calls.
- [x] #5 watchlists store includes Watchlist list/loading/error/selected state and mutator actions.
- [x] #6 Focused Vitest service/store tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing Vitest coverage for first-class Watchlist service paths and scoped child query parameters.
2. Add first-class Watchlist TypeScript types and extend source/job/fetch parameter contracts.
3. Add Watchlist CRUD service methods and thread watchlist_id through source/job/run/item/output fetchers.
4. Add watchlist_id support to overview aggregate fetches.
5. Add Watchlist list/loading/error/selected state and mutators to the Zustand store.
6. Run focused Vitest verification and update this task record.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added apps/packages/ui/src/services/__tests__/watchlists-first-class.test.ts for Watchlist CRUD service paths, watchlist_id child fetch params, typed source/job create payloads, and store container state/actions. Extended apps/packages/ui/src/services/__tests__/watchlists-overview.test.ts to verify overview aggregation forwards watchlist_id to child fetches.

Implemented WatchlistContainer, WatchlistCreate, WatchlistUpdate, domain/status/priority types, source watchlist_ids, and source/job watchlist_id fields in apps/packages/ui/src/types/watchlists.ts.

Implemented fetchWatchlists, getWatchlist, createWatchlist, updateWatchlist, deleteWatchlist, restoreWatchlist and watchlist_id query support for existing source/job/run/item/count/output fetchers in apps/packages/ui/src/services/watchlists.ts. Added watchlist_id support to OPML import fields.

Added watchlist_id parameter support to fetchWatchlistsOverviewData and threaded it through source/job/item/run/output aggregate fetches. Added watchlists/loading/error/selected container state and mutators to the Watchlists Zustand store.

Verification: bunx vitest run apps/packages/ui/src/services/__tests__/watchlists-first-class.test.ts apps/packages/ui/src/services/__tests__/watchlists-overview.test.ts -> 2 files passed, 8 tests passed. git diff --check -> clean. Bandit is not applicable because this stage only touches frontend TypeScript and Backlog task files. Attempted bunx tsc --noEmit -p apps/packages/ui/tsconfig.json, but bunx resolved TypeScript 7 and stopped on the existing tsconfig baseUrl deprecation before checking project types.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 1C adds the shared frontend Watchlist container contract: TypeScript types, API service methods, watchlist_id query threading across child fetchers and overview aggregation, and Zustand store state/actions for container selection. Focused Vitest service/overview coverage passes; full TypeScript checking was blocked by bunx resolving TypeScript 7 and hitting the existing baseUrl deprecation.
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
