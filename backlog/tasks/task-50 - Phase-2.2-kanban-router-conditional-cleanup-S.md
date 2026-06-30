---
id: TASK-50
title: Phase 2.2 kanban router conditional cleanup S
status: Done
assignee: []
created_date: '2026-05-05 01:02'
updated_date: '2026-05-05 01:19'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1273'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue #1116 Phase 2.2 by deferring the remaining kanban router imports in iter_content_router_specs while preserving route metadata and optional-import behavior. Scope is limited to the kanban endpoints block in tldw_Server_API/app/api/v1/router_groups/content.py; notes/study/persona, user content utilities, control-plane, VN assets, minimal router groups, and rag_unified remain outside this tranche.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Kanban router specs defer module import and router attribute lookup until registration or resolution
- [x] #2 Existing prefix, tags, route_key, and default_stable behavior for kanban routers remain unchanged
- [x] #3 Focused red/green router laziness coverage, full router contract tests, main router/OpenAPI contracts, Bandit touched source scan, and git diff hygiene are run before commit
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added focused router contract coverage for the nine kanban endpoint modules. The test failed red before implementation because iter_content_router_specs eagerly imported the modules and touched each router attribute during spec construction.

Converted the kanban block in content.py to lazy ImportedRouterSpec entries for kanban_boards, kanban_lists, kanban_cards, kanban_labels, kanban_checklists, kanban_comments, kanban_search, kanban_links, and kanban_workflow.

Verification: focused red failed as expected; focused green passed 1 selected; full router group contract passed 61; main router contract passed 6; OpenAPI contract suite passed 69; Bandit content router group source reported 0 results and 0 errors; git diff --check passed.

Review follow-up: collapsed the repetitive nine-entry kanban ImportedRouterSpec block into a loop over module names after Gemini review. The existing focused kanban laziness contract still covers module names, metadata, lazy imports, and attr lookup behavior.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Deferred all kanban content router registrations to lazy ImportedRouterSpec entries while preserving the /api/v1/kanban prefix, kanban tags, route_key, and default_stable behavior. Added contract coverage proving iter_content_router_specs does not import kanban modules or touch router attributes during spec construction. Addressed review feedback by using a compact loop over kanban module names.
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
