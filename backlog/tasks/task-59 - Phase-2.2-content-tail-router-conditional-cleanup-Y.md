---
id: TASK-59
title: Phase 2.2 content tail router conditional cleanup Y
status: Done
assignee:
  - codex
created_date: '2026-05-05 03:28'
updated_date: '2026-05-05 03:31'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue #1116 Phase 2.2 by deferring the adjacent tail content router imports for files, data tables, items, tasks, notifications, watchlists, integrations, scheduled tasks, and VN assets while preserving route metadata and optional-import behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 tail content router specs defer module import and router attribute lookup until registration or resolution
- [x] #2 Existing prefixes, tags, route_key values, and default_stable flags remain unchanged for files, data-tables, items, tasks, notifications, watchlists, integrations, scheduled-tasks, and vn-assets
- [x] #3 Focused red/green router laziness coverage, full router contract tests, main router/OpenAPI contracts, Bandit touched source scan, and git diff hygiene are run before commit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused failing laziness test for tail content router specs. 2. Convert files, data-tables, items, tasks, notifications, watchlists, integrations, scheduled-tasks, and vn-assets specs to ImportedRouterSpec. 3. Run focused/full router contract tests, main/OpenAPI contracts, Bandit, and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red test: pytest test_router_groups_contract.py -k tail_router_attr_lookup failed because tail fake modules had router attr access during iter_content_router_specs. Green verification: focused tail laziness test passed; full router group contract passed; main router contract passed; OpenAPI contracts passed; Bandit content.py results=0 errors=0; git diff --check clean. Documentation update not needed for this internal router import refactor.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Deferred files, data-tables, items, tasks, notifications, watchlists, integrations, scheduled-tasks, and vn-assets content router imports via ImportedRouterSpec while preserving prefixes, tags, route keys, and default_stable flags. Added focused contract coverage proving module import and router attr lookup stay lazy until resolution.
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
