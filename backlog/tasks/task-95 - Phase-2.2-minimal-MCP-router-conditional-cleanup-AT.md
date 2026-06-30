---
id: TASK-95
title: Phase 2.2 minimal MCP router conditional cleanup AT
status: Done
assignee: []
created_date: '2026-05-06 02:00'
updated_date: '2026-05-06 02:08'
labels:
  - phase2.2
  - router-cleanup
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1335'
  - 'https://github.com/rmusser01/tldw_server/pull/1337'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Convert the minimal-test mcp_unified_endpoint mcp_catalogs_manage and mcp_hub_management optional router blocks from eager try/except handling to lazy ImportedRouterSpec registration with precise optional-missing skip semantics. This unblocks later Phase 2/3 router extraction work by removing another minimal-test eager import island that would otherwise keep optional router registration coupled to import-time side effects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal mcp_unified_endpoint mcp_catalogs_manage and mcp_hub_management routers are represented as lazy ImportedRouterSpec entries.
- [x] #2 Missing target optional modules are skipped while runtime import defects propagate during registration.
- [x] #3 Existing prefixes tags route keys and default stability behavior are preserved.
- [x] #4 Focused router contract tests cover lazy attr lookup missing optional imports and runtime error propagation for this router family.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused router contract tests proving minimal MCP specs are lazy and skip only missing optional modules. 2. Replace the three eager MCP try/except blocks with ImportedRouterSpec registrations. 3. Run focused, router group, lifecycle, OpenAPI, Bandit, and diff checks. 4. Commit and open a PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: pytest test_router_groups_contract.py -k 'mcp and (attr_lookup or missing_import_failures or runtime_import_failures)' failed with three expected failures against eager MCP imports. GREEN: the same selector passed after the minimal.py change. Broader validation: router_groups_contract 113 passed, main_lifecycle_contract 54 passed, openapi_contracts 69 passed, Bandit results 0, git diff --check clean. No documentation change needed because this is an internal router registration cleanup. No known blockers.

Opened PR #1337 against dev: https://github.com/rmusser01/tldw_server/pull/1337
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted the minimal-test MCP router trio to lazy ImportedRouterSpec registration while preserving prefixes and tags. Added focused contract coverage for lazy module/attribute resolution, missing optional module skips, and runtime import defect propagation.
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
