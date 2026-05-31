---
id: TASK-349.1.2
title: Stage 4B Watchlist item batch triage and saved views API
status: Done
assignee: []
created_date: '2026-05-15 18:18'
updated_date: '2026-05-15 18:42'
labels:
  - watchlists
  - stage4
  - backend
dependencies:
  - TASK-349.1.1
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage3-content-alerts-plan.md
documentation:
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage4-review-triage-plan.md
  - Docs/API-related/Watchlists_API.md
parent_task_id: TASK-349.1
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement backend persistence and API endpoints for scalable item batch triage and per-Watchlist saved review views. Scope starts after Stage 4A query filters exist and excludes frontend integration beyond schemas/API tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A static batch item triage endpoint supports explicit item IDs and filter-scope updates under a required Watchlist scope.
- [x] #2 Batch updates can set reviewed and queued_for_briefing with deterministic matched/changed/failed/capped result summaries and bounded companion activity recording.
- [x] #3 Per-Watchlist saved review views can be created, listed, updated, and deleted with user and Watchlist scoping.
- [x] #4 Saved view payload validation rejects invalid filters, invalid sort values, and source IDs outside the selected Watchlist.
- [x] #5 Focused DB/API tests cover batch scoping, large-scope limits, saved view validation, and existing single-item PATCH compatibility.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started after Stage 4A commit 39b178da2. Scope: backend batch item triage endpoint, per-Watchlist saved review views, tests first, no frontend integration in this task.

Implemented Stage 4B backend batch item triage and saved view API. Added watchlist_item_saved_views persistence, explicit item-ID and filter-scope batch updates, source/watchlist saved-view validation, static /items/batch-update route, and nested item-view CRUD routes. Verification: Stage 4B focused DB/API tests plus adjacent Watchlists API tests passed (16 passed); Bandit touched scope passed with zero findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4B adds scalable backend support for Watchlist Updates triage: /api/v1/watchlists/items/batch-update handles selected IDs and all-filtered scopes under a required Watchlist ID, reports matched/changed/unchanged/failed/capped summaries, and records companion activity only for a bounded changed-item sample. Per-Watchlist item saved views are now persisted with CRUD endpoints and validation for sort/filter payloads and source membership. Focused and nearby backend/API tests pass, Bandit reports zero findings, and diff whitespace is clean.
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
