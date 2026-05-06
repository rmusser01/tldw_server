---
id: TASK-92
title: Phase 2.2 minimal organization router conditional cleanup AQ
status: Done
assignee: []
created_date: '2026-05-06 00:31'
updated_date: '2026-05-06 01:03'
labels:
  - phase2.2
  - router-cleanup
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1332'
  - 'https://github.com/rmusser01/tldw_server/pull/1333'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Convert the minimal-test orgs and org_invites optional router block from eager try/import handling to lazy ImportedRouterSpec registration with precise optional-missing skip semantics. This unblocks later Phase 2/3 router extraction work by removing another minimal-test eager import island that would otherwise keep optional router registration coupled to import-time side effects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal orgs and org_invites routers are represented as lazy ImportedRouterSpec entries.
- [x] #2 Missing target optional modules are skipped while runtime import defects propagate during registration.
- [x] #3 Existing prefixes tags route keys and default stability behavior are preserved.
- [x] #4 Focused router contract tests cover lazy attr lookup missing optional imports and runtime error propagation for this router family.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED router contract tests for minimal orgs/org_invites lazy import behavior, missing optional-module skips, and runtime import defect propagation. 2. Replace the eager minimal orgs and org_invites try/import blocks with lazy ImportedRouterSpec entries. 3. Run focused router tests, full router/main/OpenAPI contracts, Bandit on the touched router group, and git diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline before edits: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -q passed 101 tests. RED before production edits: focused selector -k 'minimal_optional_router_specs and org' failed on the three new organization tests because orgs and org_invites imported eagerly and did not expose named lazy specs.

Implemented orgs and org_invites as ImportedRouterSpec entries using default precise optional-missing exceptions. Verification after implementation: focused org tests 3 passed; full router group contract 104 passed; main router contract 6 passed; OpenAPI contract 69 passed; Bandit on tldw_Server_API/app/api/v1/router_groups/minimal.py reported zero findings; git diff --check passed.

Opened PR https://github.com/rmusser01/tldw_server/pull/1333 against dev for this slice.

Review follow-up: documented the later-phase unblocker explicitly. This AQ slice removes another minimal-test eager import island so later router extraction and optional-registration hardening can proceed without preserving organization import-time side effects.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Changed the minimal-test organization router tranche so orgs and org_invites are registered through lazy ImportedRouterSpec definitions instead of eager try/import blocks. This preserves prefixes and tags while narrowing skip behavior to the shared optional missing-module or missing-attribute cases, so real runtime import defects propagate during registration instead of being swallowed. Added router contract coverage for lazy attr lookup, optional missing import skips, and runtime import failure propagation. Verification: baseline router group contracts passed before edits; RED focused org tests failed before the production edit; after the edit, focused org tests passed 3/3, router group contracts passed 104/104, main router contracts passed 6/6, OpenAPI contracts passed 69/69, Bandit on minimal.py reported zero findings, and git diff --check passed.

Review follow-up: documented the later-phase unblocker for this Phase 2.2 extraction in TASK-92.
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
