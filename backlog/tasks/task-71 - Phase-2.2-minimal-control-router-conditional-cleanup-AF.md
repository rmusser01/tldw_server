---
id: TASK-71
title: Phase 2.2 minimal control router conditional cleanup AF
status: Done
assignee: []
created_date: '2026-05-05 14:25'
updated_date: '2026-05-05 14:40'
labels:
  - phase2.2
  - router-cleanup
  - issue-1116
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1301'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1116 Phase 2.2 by converting the next minimal-test optional single-router control/support registrations from eager try/import RouterSpec blocks to ImportedRouterSpec-backed lazy router specs. Scope is limited to integrations_control_plane, scheduled_tasks_control_plane, notifications, and chatbooks in tldw_Server_API/app/api/v1/router_groups/minimal.py. Preserve existing prefixes, tags, route keys, skip context, and current optional-missing skip behavior while relying on the merged fail-closed registry behavior for unexpected import failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selected minimal optional control/support router specs defer module import and router attribute lookup until registration
- [x] #2 Existing route metadata is preserved for integrations scheduled-tasks notifications and chatbooks
- [x] #3 Focused router-group test covers lazy behavior with red/green verification
- [x] #4 Router-group main-router and OpenAPI contract tests pass for the touched scope
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused lazy-import contract test for integrations_control_plane, scheduled_tasks_control_plane, notifications, and chatbooks.
2. Run the focused test red against the current eager try/import blocks.
3. Convert only those four blocks to ImportedRouterSpec while preserving metadata.
4. Rerun focused/full router contract gates, main/OpenAPI contracts, Bandit, and diff checks.
5. Review-fix tranche for PR #1305: centralize the duplicated minimal skip context for the recent data/resource and control/support blocks, add explicit default_stable=True expectations/assertions for the four converted control/support specs, then rerun the focused router contract test, full router contract suite, Bandit on minimal.py, diff check, and status check before committing/pushing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the minimal control/support router tranche. Added red/green coverage proving integrations_control_plane, scheduled_tasks_control_plane, notifications, and chatbooks defer module import and router attr lookup until ImportedRouterSpec resolution. Converted only those four eager try/import blocks in minimal.py and centralized the shared skip_context value.

Reopened after PR #1305 review feedback. Live unresolved threads: Gemini requested deduplicating the repeated minimal skip context string; CodeRabbit requested default_stable assertions for the four converted specs. Verified existing RouterSpec/ImportedRouterSpec defaults and previous eager blocks preserve default_stable=True for all four converted specs.

Review feedback addressed: minimal.py now uses one minimal_skip_context for the recent data/resource and control/support ImportedRouterSpec blocks; the control/support lazy-import contract test now asserts default_stable=True for integrations_control_plane, scheduled_tasks_control_plane, notifications, and chatbooks.

Review-fix verification: focused control/support lazy-import contract test passed; full test_router_groups_contract.py passed; Bandit on tldw_Server_API/app/api/v1/router_groups/minimal.py reported 0 results and 0 errors; git diff --check was clean before commit.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted selected minimal control/support optional router registrations to ImportedRouterSpec while preserving prefixes and tags. Verification: focused red/green test, full router_groups contract suite, main router contract suite, OpenAPI contracts, Bandit on minimal.py, and git diff --check.

PR #1305 review fix: addressed Gemini skip-context deduplication and CodeRabbit default_stable contract coverage feedback while preserving existing default_stable=True behavior.
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
