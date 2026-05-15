---
id: TASK-353
title: Implement Stage 1C Watchlist frontend contract
status: In Progress
assignee: []
created_date: '2026-05-15 01:56'
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
- [ ] #1 Frontend service tests cover Watchlist CRUD paths and watchlist_id query parameters for existing source/job/run/item/output fetchers.
- [ ] #2 watchlists.ts types expose Watchlist domain/status/priority/container/create/update contracts and backward-compatible watchlist_id fields on source/job contracts.
- [ ] #3 watchlists.ts service exposes fetch/get/create/update/delete/restore Watchlist methods and scopes child fetchers with watchlist_id when supplied.
- [ ] #4 watchlists-overview.ts accepts watchlist_id and forwards it to aggregate calls.
- [ ] #5 watchlists store includes Watchlist list/loading/error/selected state and mutator actions.
- [ ] #6 Focused Vitest service/store tests pass.
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

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
