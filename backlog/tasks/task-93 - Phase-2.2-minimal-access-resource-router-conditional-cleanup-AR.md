---
id: TASK-93
title: Phase 2.2 minimal access/resource router conditional cleanup AR
status: Done
assignee: []
created_date: '2026-05-06 01:13'
updated_date: '2026-05-06 01:23'
labels:
  - phase2.2
  - router-cleanup
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1333'
  - 'https://github.com/rmusser01/tldw_server/pull/1334'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Convert the minimal-test resource_governor and users optional router blocks from eager try/except handling to lazy ImportedRouterSpec registration with precise optional-missing skip semantics. This unblocks later Phase 2/3 router extraction work by removing another minimal-test eager import island that would otherwise keep optional router registration coupled to import-time side effects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal resource_governor and users routers are represented as lazy ImportedRouterSpec entries.
- [x] #2 Missing target optional modules are skipped while runtime import defects propagate during registration.
- [x] #3 Existing prefixes tags route keys and default stability behavior are preserved.
- [x] #4 Focused router contract tests cover lazy attr lookup missing optional imports and runtime error propagation for this router family.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED router contract tests for minimal resource_governor/users lazy import behavior, missing optional-module skips, and runtime import defect propagation. 2. Replace the eager minimal resource_governor and users try/except blocks with lazy ImportedRouterSpec entries. 3. Run focused router tests, full router group contracts, main lifecycle contracts, OpenAPI contracts, Bandit on the touched router group, and git diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline before edits: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k 'minimal_optional_router_specs and (resource or users)' -q passed 1 existing selected test. RED before production edits: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k 'access_resource' -q failed 3/3 because resource_governor and users imported eagerly, missing imports left no lazy specs, and runtime failures could not propagate during registration.

Implemented resource_governor and users as ImportedRouterSpec entries using default precise optional-missing exceptions. Verification after implementation: focused access_resource tests passed 3/3; full router group contract passed 107/107; OpenAPI contract passed 69/69; main lifecycle contract passed 54/54. A stale selector command, test_main_lifecycle_contract.py -k router -q, selected zero tests on current dev and was not counted as validation. Bandit on tldw_Server_API/app/api/v1/router_groups/minimal.py reported zero findings; git diff --check passed.

Opened PR https://github.com/rmusser01/tldw_server/pull/1334 against dev for this slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Changed the minimal-test access/resource router tranche so resource_governor and users are registered through lazy ImportedRouterSpec definitions instead of eager try/except blocks. This preserves prefixes and tags while narrowing skip behavior to the shared optional missing-module or missing-attribute cases, so real runtime import defects propagate during registration instead of being swallowed. Added router contract coverage for lazy attr lookup, optional missing import skips, and runtime import failure propagation. Verification: baseline selected router test passed before edits; RED focused access/resource tests failed before the production edit; after the edit, focused access/resource tests passed 3/3, router group contracts passed 107/107, main lifecycle contracts passed 54/54, OpenAPI contracts passed 69/69, Bandit on minimal.py reported zero findings, and git diff --check passed.

Opened PR https://github.com/rmusser01/tldw_server/pull/1334 against dev.
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
