---
id: TASK-30
title: Phase 2.2 content knowledge router conditional cleanup I
status: Done
assignee:
  - codex
created_date: '2026-05-04 03:01'
updated_date: '2026-05-04 03:05'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Defer content knowledge/query router imports for claims, text2sql, and email after PR #1250 merged, preserving existing route metadata while adding coverage for lazy router attr lookup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 claims, text2sql, and email router specs defer router attribute lookup until resolution
- [x] #2 Router prefix, tags, route_key, and default_stable metadata remain unchanged
- [x] #3 Focused router group tests, full router group tests, main router/openapi contract tests, Bandit touched source scan, and diff check are run before commit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused router group contract test that stubs claims, text2sql, and email modules with lazy __getattr__ router access and verifies iter_content_router_specs does not touch router attributes during spec construction. 2. Run the focused test before implementation and confirm it fails because the current content group eagerly resolves those routers. 3. Convert only claims, text2sql, and email to ImportedRouterSpec via append_imported_router_spec, preserving prefix, tags, route_key, and default_stable semantics. 4. Rerun the focused test, touched router group contracts, main router contract, OpenAPI contracts, Bandit on content.py, and git diff --check. 5. Update TASK-30 with verification and final summary, then commit/push/open the next PR if the tranche is clean.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red/green verified the new knowledge router laziness contract: initial focused run failed because claims, text2sql, and email resolved router attrs during iter_content_router_specs; after converting those three to ImportedRouterSpec, the focused test passed.

Verification: focused knowledge_router_attr_lookup passed; full router_groups_contract passed 50 tests; main_router_contract passed 6 tests; openapi_contracts passed 69 tests; Bandit content.py JSON reported 0 results and 0 errors; git diff --check was clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted the covered claims, text2sql, and email content router registrations from eager imports to ImportedRouterSpec-based lazy registrations. This preserves the existing route prefix, tags, route_key, and default_stable behavior while allowing route policy and registration to defer router module/attribute resolution consistently with the prior Phase 2.2 tranches. Added a focused contract test that first failed against the eager imports and now proves those router attributes are not accessed until RouterSpec resolution.
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
