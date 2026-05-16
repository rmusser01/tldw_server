---
id: TASK-355
title: Implement Stage 1D Watchlist selector shell and scoped child tabs
status: Done
assignee: []
created_date: '2026-05-15 02:02'
updated_date: '2026-05-15 03:09'
labels:
  - watchlists
  - frontend
  - ui
  - stage1d
dependencies:
  - TASK-353
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
Implement the UI shell slice from the Stage 1 first-class Watchlists plan. Scope is limited to a lightweight Watchlist selector/metadata shell on /watchlists, minimal create/edit/delete/restore controls when feasible, and threading selectedWatchlistId into existing overview/source/job/run/item/output child fetches. Do not redesign the full tabs or add Stage 2 CTI/news presets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WatchlistsPlaygroundPage loads Watchlists, selects a default/current Watchlist, and exposes a selector/metadata shell without breaking existing tab routing.
- [x] #2 User can create and edit Watchlist metadata through minimal existing UI patterns.
- [x] #3 Child tabs and overview fetch data with the selected watchlist_id where the Stage 1C services support it.
- [x] #4 Empty/loading/error states distinguish missing Watchlist containers from child-tab empty states.
- [x] #5 Extension-sized or constrained viewport behavior remains usable in focused browser/Playwright verification when the dev server is available.
- [x] #6 Focused Watchlists UI Vitest tests pass, and browser/CDP smoke evidence is recorded or a blocker is documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect WatchlistsPlaygroundPage and child tab fetch ownership.
2. Add failing component/service tests for selector loading/default selection and scoped child fetch params.
3. Add a lightweight Watchlist selector shell and minimal create/edit controls using existing Ant Design patterns.
4. Thread selectedWatchlistId into overview and child tab fetch calls without redesigning tabs.
5. Add container-level empty/loading/error handling.
6. Run focused Vitest tests and browser/CDP smoke if the WebUI can run in this worktree.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented selected Watchlist loading/default selection and a metadata shell on /watchlists, including create and edit flows. Threaded selected watchlist_id through overview, sources, jobs, runs, items, outputs, template usage, bulk import, job forms, and related polling where Stage 1C services support it. Added container-level loading, error, and empty states. Added constrained-viewport containment around the page shell and active tab area. Verification recorded: focused Watchlists Vitest component tree passed, service and route focused tests passed, git diff check passed, and Playwright/CDP smoke at 390px passed with mocked API routes and observed scoped watchlist_id requests. Documentation update was not needed for this frontend shell slice. Bandit is not applicable because touched implementation files are frontend TypeScript/React; package-wide TypeScript still has existing unrelated baseline failures outside Watchlists.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 1D adds the first-class Watchlist selector and metadata shell, minimal create/edit metadata management, selected watchlist_id scoping across child tabs, container-level states, and extension-sized viewport containment. Focused Watchlists Vitest suites, service and route tests, git diff check, and Playwright/CDP smoke passed. Package-wide TypeScript remains blocked by unrelated existing repo baseline errors.
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
