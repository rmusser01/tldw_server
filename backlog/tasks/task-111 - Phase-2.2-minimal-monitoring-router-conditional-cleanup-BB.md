---
id: TASK-111
title: Phase 2.2 minimal monitoring router conditional cleanup BB
status: Done
assignee: []
created_date: '2026-05-07 05:40'
updated_date: '2026-05-07 06:01'
labels:
  - phase-2.2
  - router-groups
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1360'
  - 'https://github.com/rmusser01/tldw_server/pull/1361'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the remaining minimal-test monitoring router factory onto the shared lazy optional-router registration path so it has the same missing-target skip semantics and diagnostics as the rest of the minimal optional router group.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal monitoring route metadata is represented through the shared optional router spec path while preserving prefix /api/v1, monitoring tag, and route_key monitoring.
- [x] #2 Focused contract coverage proves monitoring module import and router attribute lookup stay deferred until router resolution.
- [x] #3 Focused contract coverage verifies missing monitoring target skips while runtime import defects propagate.
- [x] #4 Existing router group, main router, and OpenAPI contracts still pass for the touched scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red/green evidence: focused selector tldw_Server_API/tests/Services/test_router_groups_contract.py -k 'minimal_optional_router_specs and monitoring' failed before production changes with 5 expected failures plus 1 pass, then passed after the ImportedRouterSpec change with 6 passed, 135 deselected.

Validation: router group contracts passed with 141 passed; main router contracts passed with 6 passed; OpenAPI contracts passed with 69 passed; Bandit on tldw_Server_API/app/api/v1/router_groups/minimal.py reported 0 results and 0 errors; git diff --check was clean.

Docs: no user-facing documentation update was needed for this internal router registration cleanup. Known skips/blockers: none.

Post self-review validation after loosening the skip-log assertion: focused monitoring selector passed with 6 passed, 135 deselected; git diff --check remained clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved the minimal monitoring router from a hand-written lazy factory to the shared ImportedRouterSpec registration path. This preserves the existing /api/v1 prefix, monitoring tag, and route key while using the shared missing-module/missing-attribute skip semantics and letting runtime import defects propagate.
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
