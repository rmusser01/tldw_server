---
id: TASK-91
title: Phase 2.2 minimal ops router conditional cleanup AP
status: Done
assignee: []
created_date: '2026-05-06 00:03'
updated_date: '2026-05-06 00:16'
labels:
  - phase2.2
  - router-cleanup
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1331'
  - 'https://github.com/rmusser01/tldw_server/pull/1332'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Convert the minimal-test jobs audit and config optional router block from eager try/import handling to lazy ImportedRouterSpec registration with precise optional-missing skip semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal jobs_admin audit config_info and config_admin routers are represented as lazy ImportedRouterSpec entries.
- [x] #2 Missing target optional modules are skipped while runtime import defects propagate during registration.
- [x] #3 Existing prefixes tags route keys and default stability behavior are preserved.
- [x] #4 Focused router contract tests cover lazy attr lookup missing optional imports and runtime error propagation for this router family.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED router contract tests for minimal ops lazy import behavior, missing optional-module skips, and runtime import defect propagation. 2. Replace the eager minimal jobs_admin, audit, config_info, and config_admin try/import blocks with lazy ImportedRouterSpec entries. 3. Run focused router tests, full router/main/OpenAPI contracts, Bandit on the touched router group, and git diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified RED before production edits: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k 'minimal_optional_router_specs and ops' -q failed on the three new ops tests because the current implementation imported ops routers eagerly and did not expose named lazy specs.

Implemented the minimal ops router conversion with ImportedRouterSpec entries using default precise optional-missing exceptions. Verification after implementation: focused ops tests 3 passed; full router group contract 101 passed; main router contract 6 passed; OpenAPI contract 69 passed; Bandit on tldw_Server_API/app/api/v1/router_groups/minimal.py reported zero findings; git diff --check passed.

Opened PR https://github.com/rmusser01/tldw_server/pull/1332 against dev for this slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Changed the minimal-test ops router tranche so jobs_admin, audit, config_info, and config_admin are registered through lazy ImportedRouterSpec definitions instead of eager try/import blocks. This preserves prefixes and tags while narrowing skip behavior to the shared optional missing-module or missing-attribute cases, so real runtime import defects propagate during registration instead of being swallowed. Added router contract coverage for lazy attr lookup, optional missing import skips, and runtime import failure propagation. Verification: RED focused ops tests failed before the production edit; after the edit, focused ops tests passed 3/3, router group contracts passed 101/101, main router contracts passed 6/6, OpenAPI contracts passed 69/69, Bandit on minimal.py reported zero findings, and git diff --check passed.
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
