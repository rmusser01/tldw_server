---
id: TASK-349.3
title: Plan Stage 6 Watchlist extension-sized full management
status: Done
labels:
- watchlists
- stage6
- planning
- ux
priority: medium
parent_task_id: TASK-349
documentation:
- Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the Stage 6 implementation plan for first-class Watchlists: make full Watchlist management viable in constrained/extension-sized viewports without regressing desktop flows. Scope includes route/layout audit, responsive navigation, table-to-list/detail alternatives, CRUD workflow coverage, keyboard/accessibility checks, real-server CDP smoke expectations, and child task decomposition.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stage 6 plan decomposes constrained/extension-sized Watchlists management into reviewable child tasks with clear ownership boundaries.
- [x] #2 Plan identifies exact current WebUI/shared UI files, tests, route dependencies, and real-server CDP verification paths to reuse.
- [x] #3 Plan preserves desktop behavior and existing Watchlist scoped child flows while defining mobile/list-detail alternatives for dense tabs.
- [x] #4 Plan includes TDD-first verification commands, accessibility/keyboard checks, screenshot requirements, Bandit applicability, and rollout gates.
- [x] #5 Backlog child tasks are created for each implementation slice and linked to the Stage 6 plan.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created Stage 6 implementation plan at `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md`. The plan is grounded in the approved first-class Watchlists design and current WebUI files. It decomposes constrained full management into five reviewable child tasks: navigation shell, Sources/Monitors, Activity/Reports/Templates, CRUD/accessibility hardening, and real-server CDP closeout.

Created child tasks `TASK-349.3.1` through `TASK-349.3.5`, each linked to the Stage 6 plan, approved design spec, and Stage 5 report plan.

Verification: `rg -n "TASK-349.3|Stage 6" ...` confirmed the parent plan and child tasks are discoverable. `git diff --check` passed. Bandit is not applicable because this planning slice changes Markdown/Backlog files only and no Python code.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Planned Stage 6 extension-sized full management for `/watchlists`. The plan preserves existing desktop tabs and scoped child flows while adding constrained navigation, list/detail alternatives for wide table surfaces, modal/drawer accessibility gates, and real-server CDP verification without mocked servers.
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
