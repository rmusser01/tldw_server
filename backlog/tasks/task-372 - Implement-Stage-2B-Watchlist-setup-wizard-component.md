---
id: TASK-372
title: Implement Stage 2B Watchlist setup wizard component
status: To Do
assignee: []
created_date: '2026-05-15 04:55'
labels:
  - watchlists
  - stage2
  - frontend
dependencies:
  - TASK-371
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage2-setup-wizard-plan.md
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage2-setup-wizard-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build the React setup wizard component for Stage 2 using the Stage 2A model. Scope: Ant Design wizard modal/drawer, domain preset selection, start mode selection, objective/tracked scope fields, optional source/report/monitor fields, review step, validation, and component tests with injected service callbacks. No shell integration in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 WatchlistSetupWizard renders domain presets, start modes, setup fields, and review step.
- [ ] #2 Component tests cover CTI/news preset behavior, topic-only creation, source-backed creation, report-goal creation, and required-name validation.
- [ ] #3 Component accepts service callbacks so tests do not mock the whole page shell.
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
