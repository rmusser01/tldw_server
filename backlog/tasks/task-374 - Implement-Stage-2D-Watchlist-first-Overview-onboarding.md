---
id: TASK-374
title: Implement Stage 2D Watchlist-first Overview onboarding
status: To Do
assignee: []
created_date: '2026-05-15 04:56'
labels:
  - watchlists
  - stage2
  - frontend
dependencies:
  - TASK-373
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
Reposition the existing Overview Quick Setup after the Stage 2 shell setup wizard. Scope: prevent source-first auto-open when no Watchlist is selected, frame existing quick setup as adding initial collection to the selected Watchlist, preserve pipeline builder scope, update copy/tests, and run constrained viewport CDP smoke if UI behavior changes materially.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Overview no longer bypasses Watchlist-first creation with source-first auto-open behavior.
- [ ] #2 Existing selected-Watchlist quick setup still creates scoped source/job payloads.
- [ ] #3 User-facing Overview copy frames quick setup as initial collection inside the selected Watchlist.
- [ ] #4 Constrained viewport smoke evidence is recorded if the rendered flow changes.
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
