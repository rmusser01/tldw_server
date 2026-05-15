---
id: TASK-378.3
title: Stage 3C Watchlist pipeline alert triggering and health separation
status: To Do
assignee: []
created_date: '2026-05-15 14:53'
labels:
  - watchlists
  - stage3
  - pipeline
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
Trigger content alerts from the Watchlists ingestion pipeline and keep run-stat alert-rule notifications health-oriented.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Newly recorded matching items create deduped content alerts with item evidence.
- [ ] #2 Pipeline content-alert evaluation is non-critical and cannot fail a scrape run.
- [ ] #3 Run-stat alert rules remain backward compatible while using health-oriented notification type or metadata.
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
