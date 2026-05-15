---
id: TASK-378.2
title: Stage 3B Watchlist content alert API and docs
status: To Do
assignee: []
created_date: '2026-05-15 14:53'
labels:
  - watchlists
  - stage3
  - api
  - docs
dependencies: []
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage3-content-alerts-plan.md
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
documentation:
  - Docs/API-related/Watchlists_API.md
parent_task_id: TASK-378
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose nested content alert rule and alert inbox endpoints for selected Watchlists and document the content-alert versus health-rule boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Nested Watchlist content alert APIs support rule CRUD, alert list/detail, filters, and review-state updates.
- [ ] #2 API validation returns clear errors for invalid regex, source constraints, and missing Watchlist scope.
- [ ] #3 Docs distinguish content alerts from health rules and identify Topic Monitoring as an internal dependency boundary.
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
