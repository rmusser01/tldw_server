---
id: TASK-373
title: Implement Stage 2C Watchlist setup shell integration
status: To Do
assignee: []
created_date: '2026-05-15 04:56'
labels:
  - watchlists
  - stage2
  - frontend
dependencies:
  - TASK-372
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
Integrate the Stage 2 setup wizard into the Watchlists shell create flow. Scope: replace create-mode modal with setup wizard, keep edit modal for metadata edits, use existing service functions to create Watchlist/source/job with watchlist_id propagation, select the created Watchlist, and route to the correct scoped tab. No Overview quick setup repositioning in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Primary create control opens the Stage 2 setup wizard while edit keeps metadata modal behavior.
- [ ] #2 Topic-only completion creates/selects a Watchlist and routes to Feeds or the planned destination.
- [ ] #3 Source-backed/report-goal completion sends watchlist_id on source and job payloads.
- [ ] #4 Existing selected-scope route/service tests remain green.
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
