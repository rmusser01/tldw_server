---
id: TASK-371
title: Implement Stage 2A Watchlist setup wizard model
status: To Do
assignee: []
created_date: '2026-05-15 04:54'
labels:
  - watchlists
  - stage2
  - frontend
dependencies:
  - TASK-370
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
Build the Watchlist setup model for Stage 2. Scope: domain/start-mode types, CTI OSINT/news/general/blank presets, payload builders for Watchlist/source/monitor setup, source URL normalization, and copy contract tests. No React component or shell wiring in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Setup wizard model helpers exist with typed presets and start modes.
- [ ] #2 Helper tests cover CTI/news/general/blank presets, topic-only no-source path, source-backed path, report-goal path, and URL normalization.
- [ ] #3 Copy contract covers Stage 2 preset/start-mode labels and preserves Stage 3 alert boundary.
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
