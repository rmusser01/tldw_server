---
id: TASK-52
title: Phase 2.2 notes router conditional cleanup T
status: Done
assignee:
  - codex
created_date: '2026-05-05 01:39'
updated_date: '2026-05-05 01:43'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1276'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue #1116 Phase 2.2 after PR #1276 by deferring the notes-focused content router imports in iter_content_router_specs while preserving route metadata and optional-import behavior. Scope is limited to notes_graph, notes, and web_clipper in tldw_Server_API/app/api/v1/router_groups/content.py.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 notes_graph, notes, and web_clipper router specs defer module import and router attribute lookup until registration or resolution
- [x] #2 Existing prefixes, tags, route_key values, and default_stable behavior for the notes tranche remain unchanged
- [x] #3 Focused red/green router laziness coverage, full router contract tests, main router/OpenAPI contracts, Bandit touched source scan, and git diff hygiene are run before commit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused router contract coverage showing iter_content_router_specs does not import or touch router attrs for notes_graph, notes, and web_clipper during spec construction. 2. Run the focused test red against origin/dev behavior. 3. Replace only the notes-focused try/except blocks with ImportedRouterSpec entries via append_imported_router_spec. 4. Re-run focused/full router contracts, main router contracts, OpenAPI contracts, Bandit on content.py, and diff hygiene. 5. Commit and open/update the next PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added focused red/green router contract coverage for notes_graph, notes, and web_clipper. Red failed on origin/dev because iter_content_router_specs eagerly resolved all three router attributes during spec construction. Replaced only the notes-focused try/except blocks in content.py with ImportedRouterSpec entries while preserving notes_graph ordering before generic notes and keeping prefixes, tags, route keys, and default_stable behavior unchanged. Verification: focused notes laziness test red then green; router group contracts 62 passed; main router contract 6 passed; OpenAPI contracts 69 passed; Bandit touched source reported 0 results and 0 errors; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Deferred notes_graph, notes, and web_clipper content router registrations to lazy ImportedRouterSpec entries while preserving route metadata and optional-import behavior. Added contract coverage proving iter_content_router_specs does not resolve those router attrs during spec construction.
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
