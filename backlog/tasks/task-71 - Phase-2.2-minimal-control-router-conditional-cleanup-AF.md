---
id: TASK-71
title: Phase 2.2 minimal control router conditional cleanup AF
status: Done
assignee: []
created_date: '2026-05-05 14:25'
updated_date: '2026-05-05 14:28'
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
1. Add a focused lazy-import contract test for integrations_control_plane, scheduled_tasks_control_plane, notifications, and chatbooks. 2. Run the focused test red against the current eager try/import blocks. 3. Convert only those four blocks to ImportedRouterSpec while preserving metadata. 4. Rerun focused/full router contract gates, main/OpenAPI contracts, Bandit, and diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the minimal control/support router tranche. Added red/green coverage proving integrations_control_plane, scheduled_tasks_control_plane, notifications, and chatbooks defer module import and router attr lookup until ImportedRouterSpec resolution. Converted only those four eager try/import blocks in minimal.py and centralized the shared skip_context value.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted selected minimal control/support optional router registrations to ImportedRouterSpec while preserving prefixes and tags. Verification: focused red/green test, full router_groups contract suite, main router contract suite, OpenAPI contracts, Bandit on minimal.py, and git diff --check.
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
