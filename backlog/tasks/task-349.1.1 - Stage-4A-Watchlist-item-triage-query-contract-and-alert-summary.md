---
id: TASK-349.1.1
title: Stage 4A Watchlist item triage query contract and alert summary
status: To Do
assignee: []
created_date: '2026-05-15 18:18'
labels:
  - watchlists
  - stage4
  - backend
dependencies: []
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
Implement the backend/API query contract for server-authoritative Watchlist item review queues. Scope is limited to item list sorting/filtering and optional content-alert summary enrichment for the existing /api/v1/watchlists/items list; do not add batch updates, saved views, or frontend UI in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 GET /api/v1/watchlists/items supports validated server-side sort modes while preserving the current default ordering for existing clients.
- [ ] #2 Item list filtering supports content-alert context such as alert presence, alert status, alert severity, and alert rule ID without duplicating item rows.
- [ ] #3 Optional item alert summary returns compact content-alert context for each item only when requested.
- [ ] #4 Focused DB and API tests cover sort stability, Watchlist/user scoping, alert-aware filters, static route ordering, and no fake confidence/novelty filters.
- [ ] #5 Backward compatibility tests for existing first-class Watchlist and Stage 3 content-alert APIs continue to pass.
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
