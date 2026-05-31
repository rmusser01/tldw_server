---
id: TASK-370
title: Plan Stage 2 Watchlist creation and setup wizard
status: Done
assignee: []
created_date: '2026-05-15 04:48'
updated_date: '2026-05-15 04:52'
labels:
  - watchlists
  - stage2
  - planning
dependencies:
  - TASK-365
references:
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage2-setup-wizard-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a Stage 2 implementation plan for Watchlist-first onboarding. Scope: domain presets for CTI/OSINT, news, general, and blank; capture Watchlist objective/tracked scope before sources; support start from sources, start from topic, and start from report goal; define tests for no-source wizard path and constrained viewport behavior. No feature code in this task beyond the plan and tracking updates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stage 2 plan maps existing quick setup/pipeline code and the Stage 1 Watchlist container contract.
- [x] #2 Plan decomposes Stage 2 into reviewable implementation tasks with file ownership, tests, verification commands, and commit boundaries.
- [x] #3 Plan preserves Stage 3 content-match alerts and defensible report-builder work as explicit future boundaries.
- [x] #4 Backlog task records plan path, review notes, and final summary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created Stage 2 setup wizard implementation plan at Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage2-setup-wizard-plan.md. The plan maps current Watchlists shell create modal, Overview Quick Setup, Pipeline Builder, Zustand selected Watchlist state, existing service CRUD, locale mirrors, and Stage 1 scoping. It decomposes implementation into Stage 2A-2E with exact file ownership, tests, verification commands, and commit boundaries. Self-review fixed a TypeScript test-snippet issue around checking absence of alert rules. Stage 3 content-match alerts and Stage 5 defensible report-builder work remain explicit future boundaries. Verification for this planning-only slice: git diff --check passed. Bandit not run because no backend/code behavior changed in this task.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Stage 2 Watchlist-first setup wizard implementation plan. The next executable step is Task 0 in the plan: create Stage 2A-2E Backlog implementation tasks, run the current focused frontend baseline, then start Stage 2A helper/model tests.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
