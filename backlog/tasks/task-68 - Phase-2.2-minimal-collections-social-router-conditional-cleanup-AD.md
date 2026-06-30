---
id: TASK-68
title: Phase 2.2 minimal collections/social router conditional cleanup AD
status: Done
assignee: []
created_date: '2026-05-05 05:45'
updated_date: '2026-05-05 05:47'
labels:
  - phase2.2
  - router-cleanup
  - issue-1116
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1298'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1116 Phase 2.2 by converting the next remaining minimal-test optional router registrations from eager try/import RouterSpec blocks to ImportedRouterSpec-backed lazy router specs. Scope is limited to collections_feeds, collections_websub router and callback_router, slack, discord, and telegram in tldw_Server_API/app/api/v1/router_groups/minimal.py. Preserve existing prefixes, tags, route keys, skip context, and current skip-on-import-failure behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selected collections and social minimal optional router specs defer module import and router attribute lookup until registration
- [x] #2 Existing route metadata is preserved for collections-feeds collections-websub slack discord and telegram
- [x] #3 Focused router-group test covers the lazy behavior with red/green verification
- [x] #4 Router-group main-router and OpenAPI contract tests pass for the touched scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a narrow minimal collections/social router tranche. Added red/green router-group coverage that proves collections_feeds, collections_websub router, collections_websub callback_router, slack, discord, and telegram defer module import and router attribute access until ImportedRouterSpec resolution. Replaced only those eager try/import RouterSpec blocks in minimal.py.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted the selected minimal collections/social optional router registrations to ImportedRouterSpec while preserving prefixes and tags. Verification: focused red/green test, full router_groups contract suite, main router contract suite, OpenAPI contracts, Bandit on minimal.py, and git diff --check.
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
