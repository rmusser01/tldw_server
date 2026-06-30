---
id: TASK-57
title: Phase 2.2 chatbooks sharing router conditional cleanup W
status: Done
assignee:
  - codex
created_date: '2026-05-05 02:49'
updated_date: '2026-05-05 02:52'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue #1116 Phase 2.2 by deferring the adjacent chatbooks and sharing content router imports in iter_content_router_specs while preserving route metadata and optional-import behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 chatbooks and sharing router specs defer module import and router attribute lookup until registration or resolution
- [x] #2 Existing prefixes, tags, route_key values, and default_stable behavior for chatbooks and sharing remain unchanged
- [x] #3 Focused red/green router laziness coverage, full router contract tests, main router/OpenAPI contracts, Bandit touched source scan, and git diff hygiene are run before commit
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added focused red/green coverage for chatbooks and sharing router specs. Red failed on origin/dev because both fake router attrs were touched during iter_content_router_specs construction.

Converted only chatbooks and sharing from eager try/import blocks to ImportedRouterSpec entries via append_imported_router_spec, preserving prefix /api/v1, tags, route_key values, and default_stable=True behavior.

Verification passed: focused chatbooks/sharing laziness test; full router group contracts 65 passed; main router contracts 6 passed; OpenAPI contracts 69 passed; Bandit content.py 0 results and 0 errors; git diff --check.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Deferred chatbooks and sharing content router registrations to lazy ImportedRouterSpec entries while preserving route metadata and optional resolution behavior. Added contract coverage proving iter_content_router_specs does not resolve those router attrs during spec construction.
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
