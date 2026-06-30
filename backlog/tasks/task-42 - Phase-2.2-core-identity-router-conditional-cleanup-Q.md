---
id: TASK-42
title: Phase 2.2 core identity router conditional cleanup Q
status: Done
assignee: []
created_date: '2026-05-04 14:46'
updated_date: '2026-05-04 14:51'
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
Continue #1116 Phase 2.2 by deferring remaining small core identity/config/sync router imports from iter_core_router_specs while preserving route metadata and optional-import behavior. Scope is limited to auth, authnz_debug, users, user_keys, feedback, config_info, and sync; larger/minimal/content groups remain outside this tranche.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 auth, authnz_debug, users, user_keys, feedback, config_info, and sync core router specs defer router attribute lookup until registration/resolution
- [x] #2 Existing prefix, tags, route_key, and authnz_debug default_stable behavior remain unchanged
- [x] #3 Focused red/green router laziness coverage, full router contract tests, main router/OpenAPI contracts, Bandit touched source scan, and git diff hygiene are run before commit
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Starting from stacked branch codex/phase2-2-core-auth-router-conditionals-q at PR #1267 commit 4e5d09361f; this branch will be rebased/PR'd after #1267 lands or kept stacked until then.

Verification: baseline full router group contract passed 57 before edits. Red focused identity/config/sync laziness test failed before implementation because scoped router attrs were eagerly resolved. Green focused rerun passed 1 selected; full router group contract passed 58; main router contract passed 6; OpenAPI contract suite passed 69; Bandit core router group source reported 0 results and 0 errors; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted auth, authnz_debug, users, user_keys, feedback, config_info, and sync core router registrations to lazy ImportedRouterSpec entries while preserving prefixes, tags, route keys, and authnz_debug default_stable behavior. Added contract coverage proving iter_core_router_specs does not touch those router attributes during spec construction.
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
