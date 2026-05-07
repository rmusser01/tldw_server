---
id: TASK-114
title: Phase 2.2 minimal media router conditional cleanup BC
status: Done
assignee: []
created_date: '2026-05-07 14:37'
updated_date: '2026-05-07 14:45'
labels:
  - phase-2.2
  - router-groups
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1362'
  - 'https://github.com/rmusser01/tldw_server/pull/1363'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the remaining minimal-test media router factory onto the shared lazy optional-router registration path so it uses the same missing-target skip semantics and diagnostics as other optional minimal router specs while preserving the existing media route metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal media route metadata is represented through the shared optional router spec path while preserving prefix /api/v1/media, media tag, and route_key media.
- [x] #2 Focused contract coverage proves media module import and router attribute lookup stay deferred until router resolution.
- [x] #3 Focused contract coverage verifies missing media target skips while runtime import defects propagate.
- [x] #4 Existing router group, main router, and OpenAPI contracts still pass for the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED contract tests for minimal media lazy resolution, missing optional-target skips, and runtime defect propagation.
2. Move minimal media registration from a hand-written factory to ImportedRouterSpec.
3. Run focused, router-group, main-router, OpenAPI, Bandit, and diff hygiene verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "minimal_optional_router_specs and media" -q failed with expected StopIteration because no shared media ImportedRouterSpec existed yet.

GREEN/validation: focused selector passed 7 passed; router_groups_contract.py passed 153 passed; test_main_router_contract.py passed 6 passed; test_openapi_contracts.py passed 69 passed; Bandit on tldw_Server_API/app/api/v1/router_groups/minimal.py reported 0 results/errors; git diff --check passed.

No documentation update was required for this internal router-registration cleanup. No skips or blockers remain.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved the minimal media router from a hand-written lazy factory to the shared ImportedRouterSpec registration path. This preserves /api/v1/media, the media tag, route_key=media, and the minimal skip context while using the shared missing-module/missing-attribute skip semantics and allowing runtime import defects to propagate. Added focused contract coverage for lazy resolution, optional missing-target skips, missing router attribute skips, and runtime import failure propagation.
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
