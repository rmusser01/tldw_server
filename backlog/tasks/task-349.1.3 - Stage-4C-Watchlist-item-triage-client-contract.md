---
id: TASK-349.1.3
title: Stage 4C Watchlist item triage client contract
status: To Do
assignee: []
created_date: '2026-05-15 18:18'
labels:
  - watchlists
  - stage4
  - frontend
dependencies:
  - TASK-349.1.1
  - TASK-349.1.2
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
Add the frontend TypeScript and service contract for Stage 4 item triage after the backend sort/filter, batch, and saved-view APIs exist. Scope is limited to types, service methods, query serialization, and saved-view migration helpers; do not redesign ItemsTab in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Frontend Watchlists types expose item alert summaries, item sort/filter literals, batch update payloads/results, and saved view contracts.
- [ ] #2 watchlists.ts service methods serialize Stage 4 item filters/sort, batch triage requests, and saved view CRUD routes correctly.
- [ ] #3 ItemsTab utility helpers normalize saved views and preserve a recoverable localStorage-to-server migration path.
- [ ] #4 Focused Vitest service and utility tests cover query serialization, invalid localStorage data, Watchlist-scoped saved view payloads, and batch endpoint calls.
- [ ] #5 No visible ItemsTab behavior is changed outside type/service/helper wiring in this task.
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
