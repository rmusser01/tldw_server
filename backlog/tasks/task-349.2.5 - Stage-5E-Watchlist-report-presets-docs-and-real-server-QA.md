---
id: TASK-349.2.5
title: Stage 5E Watchlist report presets docs and real-server QA
status: To Do
assignee: []
created_date: '2026-05-15 21:40'
labels:
  - watchlists
  - stage5
  - docs
  - qa
dependencies:
  - TASK-349.2.4
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage4-review-triage-plan.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md
  - Docs/API-related/Watchlists_API.md
  - Docs/Published/API-related/Watchlists_API.md
parent_task_id: TASK-349.2
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close out Stage 5 by adding or updating report presets, documenting the API and user-facing evidence/readiness contract, running focused backend/frontend verification, running Bandit on touched backend code, and completing a real-server CDP smoke through /watchlists without server mocks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 CTI/OSINT and news report presets render evidence/readiness/source context from the Stage 5 output context while preserving Markdown/HTML/Chatbook/audio compatibility.
- [ ] #2 API docs and published docs describe Stage 5 output fields, evidence/readiness endpoints, metadata, warning codes, legacy behavior, and CTI/news examples.
- [ ] #3 Focused backend and frontend tests for Stage 5 plus existing output/triage regressions pass or any failures are documented with blockers.
- [ ] #4 Bandit is run on touched backend scope and no new findings are introduced.
- [ ] #5 Real FastAPI plus real WebUI CDP smoke covers CTI and news report creation, evidence inspection, preview/download, and constrained viewport management with screenshots and notes recorded.
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
