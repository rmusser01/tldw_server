---
id: TASK-69
title: Phase 2.2 minimal data/resource router conditional cleanup AE
status: Done
assignee: []
created_date: '2026-05-05 14:00'
updated_date: '2026-05-05 14:12'
labels:
  - phase2.2
  - router-cleanup
  - issue-1116
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1300'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1116 Phase 2.2 by converting the next remaining minimal-test optional router registrations from eager try/import RouterSpec blocks to ImportedRouterSpec-backed lazy router specs. Scope is limited to files, storage, data_tables, reading_highlights, items, and reminders in tldw_Server_API/app/api/v1/router_groups/minimal.py. Preserve existing prefixes, tags, route keys, skip context, and current skip-on-import-failure behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selected data/resource minimal optional router specs defer module import and router attribute lookup until registration
- [x] #2 Existing route metadata is preserved for files storage data-tables reading-highlights items and tasks
- [x] #3 Focused router-group test covers the lazy behavior with red/green verification
- [x] #4 Router-group main-router and OpenAPI contract tests pass for the touched scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Addressed PR #1301 review feedback by introducing data_resource_skip_context for the shared data/resource ImportedRouterSpec skip context.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted the selected minimal data/resource optional router registrations to ImportedRouterSpec while preserving prefixes and tags. Review feedback addressed by centralizing the repeated data/resource skip_context value. Verification: focused red/green test, focused review-fix rerun, full router_groups contract suite, main router contract suite, OpenAPI contracts, Bandit on minimal.py, and git diff --check.
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
