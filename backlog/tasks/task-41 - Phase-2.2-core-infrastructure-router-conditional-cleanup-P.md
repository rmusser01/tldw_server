---
id: TASK-41
title: Phase 2.2 core infrastructure router conditional cleanup P
status: Done
assignee: []
created_date: '2026-05-04 07:01'
updated_date: '2026-05-04 07:06'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
documentation:
  - >-
    Docs/superpowers/plans/2026-05-03-phase2-followup-stack-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue #1116 Phase 2.2 by deferring basic core infrastructure router imports from iter_core_router_specs while preserving existing route metadata and optional-import behavior. Scope is limited to health, moderation, monitoring, metrics, audit, consent, setup, and tools in router_groups/core.py; auth/user/config/sync routes remain outside this tranche.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 health, moderation, monitoring, metrics, audit, consent, setup, and tools core router specs defer router attribute lookup until registration/resolution.
- [x] #2 Existing prefix, tags, route_key, and tools default_stable behavior for the scoped core infrastructure routes remain unchanged.
- [x] #3 Focused red/green router laziness coverage, full router contract tests, main router/OpenAPI contracts, Bandit touched source scan, and git diff hygiene are run before commit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: red focused infrastructure laziness test failed before implementation; green focused rerun passed 1 selected; full router group contract passed 57; main router contract passed 6; OpenAPI contract suite passed 69; Bandit core router group source reported 0 results and 0 errors; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted health, moderation, monitoring, metrics, audit, consent, setup, and tools core router registrations to lazy ImportedRouterSpec entries while preserving prefixes, tags, route keys, and tools default_stable=false. Added contract coverage proving iter_core_router_specs does not touch those router attributes during spec construction.
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
