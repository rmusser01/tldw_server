---
id: TASK-350
title: Plan Stage 1 first-class Watchlist container implementation
status: Done
assignee: []
created_date: '2026-05-15 01:19'
updated_date: '2026-05-15 01:24'
labels:
  - watchlists
  - planning
  - ux
dependencies:
  - TASK-349
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a detailed implementation plan for Stage 1 of the approved first-class Watchlists design. The plan must decompose the Watchlist container contract, migration/default Watchlist behavior, API CRUD, child scoping compatibility, frontend shell/state changes, tests, and rollout gates into reviewable implementation tasks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan is saved under Docs/superpowers/plans/ with the required implementation-plan header and references the approved spec.
- [x] #2 Plan maps backend schema/API/frontend files to concrete responsibilities and safe boundaries.
- [x] #3 Plan decomposes Stage 1 into bite-sized tasks with test-first steps, exact commands, expected results, and commit checkpoints.
- [x] #4 Plan covers migration/default Watchlist behavior and existing endpoint compatibility for sources/jobs/runs/items/outputs.
- [x] #5 Plan includes verification, Bandit expectations for implementation, rollout gates, and known risks/open questions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan saved at `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage1-implementation-plan.md`. The plan references the approved design spec, maps backend/frontend/test responsibilities, decomposes Stage 1 into test-first implementation tasks, and defines verification gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created a Stage 1 implementation plan for first-class Watchlist containers. The plan keeps Stage 1 scoped to container persistence, default migration/backfill, CRUD API, child endpoint scoping, output provenance, frontend selector shell, focused tests, and rollout gates.

Verified plan hygiene with `git diff --check -- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage1-implementation-plan.md` and a trailing-whitespace awk check; both exited 0 with no output.

Bandit not run because this task changes only documentation and Backlog task records. The plan includes Bandit commands for the later backend implementation scope.

Subagent review was not used because the current instructions only allow subagents when explicitly authorized by the user; I performed a local review against the acceptance criteria instead.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a detailed Stage 1 implementation plan for first-class Watchlist containers. The plan translates the approved Watchlists design into concrete backend, API, frontend, test, verification, and rollout work while preserving existing `/watchlists` flows and legacy route behavior. It defines the default migrated Watchlist behavior, child scoping compatibility for sources/jobs/runs/items/outputs, output provenance expectations, frontend selector shell integration, and implementation checkpoints.

Verification: `git diff --check -- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage1-implementation-plan.md` and a trailing-whitespace awk check both passed with no output. Bandit was skipped as docs-only, with implementation Bandit expectations captured in the plan.
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
