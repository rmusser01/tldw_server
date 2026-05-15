---
id: TASK-349.1.2
title: Stage 4B Watchlist item batch triage and saved views API
status: To Do
assignee: []
created_date: '2026-05-15 18:18'
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
- [ ] #1 A static batch item triage endpoint supports explicit item IDs and filter-scope updates under a required Watchlist scope.
- [ ] #2 Batch updates can set reviewed and queued_for_briefing with deterministic matched/changed/failed/capped result summaries and bounded companion activity recording.
- [ ] #3 Per-Watchlist saved review views can be created, listed, updated, and deleted with user and Watchlist scoping.
- [ ] #4 Saved view payload validation rejects invalid filters, invalid sort values, and source IDs outside the selected Watchlist.
- [ ] #5 Focused DB/API tests cover batch scoping, large-scope limits, saved view validation, and existing single-item PATCH compatibility.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
