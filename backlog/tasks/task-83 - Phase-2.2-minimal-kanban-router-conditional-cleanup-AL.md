---
id: TASK-83
title: Phase 2.2 minimal kanban router conditional cleanup AL
status: Done
assignee: []
created_date: '2026-05-05 18:55'
updated_date: '2026-05-05 18:59'
labels:
  - phase2.2
  - router-cleanup
  - issue-1116
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1322'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1116 Phase 2.2 after PR #1322. Convert the minimal-test Kanban optional router family in tldw_Server_API/app/api/v1/router_groups/minimal.py from one eager broad try/import RouterSpec block to ImportedRouterSpec-backed lazy router specs. Scope is limited to kanban_boards, kanban_lists, kanban_cards, kanban_labels, kanban_checklists, kanban_comments, kanban_search, kanban_links, and kanban_workflow. Preserve prefix /api/v1/kanban, tags, route_key=kanban, default_stable behavior, and existing minimal-test broad skip behavior for import-time failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Kanban minimal optional specs defer module import and router attribute lookup until registration
- [x] #2 Existing route metadata is preserved for every Kanban router in scope
- [x] #3 Focused router-group tests cover lazy behavior and broad import failure skipping with red-green verification
- [x] #4 Router-group contract tests and touched-source Bandit pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Red-green Kanban tranche: add focused router-group tests for lazy minimal Kanban module import/router attribute lookup and broad runtime import-failure skipping; then replace only the eager minimal Kanban try/import block with ImportedRouterSpec entries preserving prefix, tags, route_key, default_stable, and minimal skip context.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red verification: focused minimal Kanban router tests failed before production changes because Kanban endpoint modules were imported during spec construction and named lazy specs were absent.

Green verification: focused Kanban tests, full router group contract tests, main router contract tests, touched-source Bandit, and git diff --check all passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted the minimal-test Kanban optional router family to ImportedRouterSpec-backed lazy specs. The new entries preserve /api/v1/kanban prefix, kanban tags, route_key=kanban, default_stable behavior, and broad minimal-test skip semantics with skip_exceptions=(Exception,) while deferring module import and router attribute lookup until registration. Added regression coverage for lazy Kanban resolution and RuntimeError skip behavior. Verification: focused red/green Kanban tests, full test_router_groups_contract.py, test_main_router_contract.py, Bandit on minimal.py with 0 results/0 errors, and git diff --check.
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
