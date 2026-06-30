---
id: TASK-61
title: Phase 2.2 RAG router conditional cleanup Z
status: Done
assignee:
  - codex
created_date: '2026-05-05 03:50'
updated_date: '2026-05-05 03:53'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete #1116 Phase 2.2 content router cleanup by deferring the remaining rag_unified content router import while preserving route metadata and optional-import behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 rag_unified content router spec defers module import and router attribute lookup until registration or resolution
- [x] #2 Existing tags, route_key, prefix, and default_stable behavior remain unchanged for the rag-unified router
- [x] #3 Focused red/green router laziness coverage, full router contract tests, main router/OpenAPI contracts, Bandit touched source scan, and git diff hygiene are run before commit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused failing laziness test for rag_unified. 2. Convert rag_unified to ImportedRouterSpec. 3. Run focused/full router contract tests, main/OpenAPI contracts, Bandit, and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red test: pytest test_router_groups_contract.py -k rag_unified_router_attr_lookup failed because rag_unified fake module had router attr access during iter_content_router_specs. Green verification: focused RAG laziness test passed; full router group contract passed; main router contract passed; OpenAPI contracts passed; Bandit content.py results=0 errors=0; git diff --check clean. Documentation update not needed for this internal router import refactor.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Deferred the remaining rag_unified content router import via ImportedRouterSpec while preserving tags, route_key, prefix, and default_stable behavior. Added focused contract coverage proving module import and router attr lookup stay lazy until resolution.
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
