---
id: TASK-63
title: Phase 2.2 minimal RAG router conditional cleanup AA
status: Done
assignee: []
created_date: '2026-05-05 04:09'
updated_date: '2026-05-05 04:12'
labels:
  - phase2.2
  - router-cleanup
  - issue-1116
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1291'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1116 Phase 2.2 by converting the minimal-test app RAG optional router registrations for rag_unified and rag_health from eager try/import RouterSpec blocks to ImportedRouterSpec-backed lazy router specs. Keep this tranche intentionally narrow and preserve existing prefixes, tags, route keys, and skip behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 minimal rag_unified and rag_health optional router specs defer module import and router attribute lookup until registration
- [x] #2 existing route metadata for rag_unified and rag_health is preserved
- [x] #3 focused router-group tests cover the lazy behavior with red/green verification
- [x] #4 router-group, main-router, and OpenAPI contract tests pass for the touched scope
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused red test for minimal RAG optional router laziness. 2. Convert only rag_unified and rag_health minimal optional router blocks to ImportedRouterSpec. 3. Run focused, router-group, main-router, OpenAPI, Bandit, and diff hygiene checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red: pytest test_router_groups_contract.py -k minimal_optional_router_specs_defers_rag_attr_lookup failed because rag_unified.router and rag_health.router were resolved during iter_minimal_optional_router_specs(). Green: focused test passed after converting those two specs. Full touched gates passed: router_groups_contract 69 passed; main_router_contract 6 passed; openapi_contracts 69 passed; Bandit minimal.py results=0 errors=0; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted the minimal-test app rag_unified and rag_health optional router registrations to ImportedRouterSpec-backed lazy specs while preserving empty prefixes, tags, route keys, and minimal skip context. Added focused contract coverage proving module import and router attr lookup stay deferred until spec resolution.
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
