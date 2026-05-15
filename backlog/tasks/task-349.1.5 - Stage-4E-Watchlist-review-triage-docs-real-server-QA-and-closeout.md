---
id: TASK-349.1.5
title: Stage 4E Watchlist review triage docs real-server QA and closeout
status: To Do
assignee: []
created_date: '2026-05-15 18:19'
labels:
  - watchlists
  - stage4
  - qa
dependencies:
  - TASK-349.1.1
  - TASK-349.1.2
  - TASK-349.1.3
  - TASK-349.1.4
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
Close Stage 4 by documenting the item triage API/UI contract, running focused backend/frontend verification, running Bandit for touched backend code, and performing a real-server CDP smoke of the refreshed Updates triage flow in desktop and extension-sized viewports. Do not mock the server for browser evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Watchlists API docs describe Stage 4 item sort/filter additions, alert summaries, batch triage, saved views, and the Stage 5 report-builder boundary.
- [ ] #2 Focused backend and frontend Stage 4 test suites pass or any scoped failures are fixed/documented with concrete evidence.
- [ ] #3 Bandit runs against touched backend files and results are recorded.
- [ ] #4 Real FastAPI plus real WebUI CDP smoke covers desktop and extension-sized Updates triage flows without API mocking.
- [ ] #5 Stage 4 Backlog tasks are closed or left with explicit blockers and final summaries.
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
