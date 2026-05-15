---
id: TASK-378.1
title: Stage 3A Watchlist content alert persistence and matcher
status: To Do
assignee: []
created_date: '2026-05-15 14:53'
labels:
  - watchlists
  - stage3
  - backend
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
Implement Watchlists-owned content alert persistence and deterministic matcher service for first-class Watchlists Stage 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Content alert rule and alert records are persisted in the Watchlists data model without overloading run-stat alert rules.
- [ ] #2 Matcher creates evidence-backed alerts for matching newly collected Watchlist items.
- [ ] #3 Focused DB and pipeline matcher tests cover scoping, validation, dedupe, source constraints, and non-critical failure handling.
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
