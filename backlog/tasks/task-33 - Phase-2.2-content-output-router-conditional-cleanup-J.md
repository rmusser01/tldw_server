---
id: TASK-33
title: Phase 2.2 content output router conditional cleanup J
status: Done
assignee:
  - codex
created_date: '2026-05-04 03:44'
updated_date: '2026-05-04 03:47'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Defer the covered outputs_templates and outputs content router imports after PR #1255 merged, preserving existing route metadata and adding lazy router attribute coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 outputs_templates and outputs router specs defer router attribute lookup until resolution
- [x] #2 Router prefix, tags, route_key, and default_stable metadata remain unchanged
- [x] #3 Focused router group tests, full router group tests, main router/openapi contract tests, Bandit touched source scan, and diff check are run before commit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused router group contract test that stubs outputs_templates and outputs modules with lazy __getattr__ router access and verifies iter_content_router_specs does not touch router attributes during spec construction. 2. Run the focused test before implementation and confirm it fails against the current eager imports. 3. Convert only outputs_templates and outputs to ImportedRouterSpec via append_imported_router_spec, preserving prefix, tags, route_key, and default_stable semantics. 4. Rerun the focused test, full router group contract file, main router contract, OpenAPI contracts, Bandit on content.py, and git diff --check. 5. Update TASK-33 with verification and final summary, then commit/push/open the next PR if clean.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red/green verified the output router laziness contract: initial focused run failed because outputs_templates and outputs resolved router attrs during iter_content_router_specs; after converting both to ImportedRouterSpec, the focused test passed.

Verification: focused output_router_attr_lookup passed; full router_groups_contract passed 51 tests; main_router_contract passed 6 tests; openapi_contracts passed 69 tests; Bandit content.py JSON reported 0 results and 0 errors; git diff --check was clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted the covered outputs_templates and outputs content router registrations from eager imports to ImportedRouterSpec-based lazy registrations. This preserves the existing prefix, tags, route_key, and default_stable behavior while continuing the Phase 2.2 route-policy-friendly lazy registration pattern. Added a focused contract test that first failed against the eager imports and now proves both output router attributes are not accessed until RouterSpec resolution.
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
