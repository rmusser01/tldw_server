---
id: TASK-65
title: Phase 2.2 minimal embedding router conditional cleanup AB
status: Done
assignee: []
created_date: '2026-05-05 05:14'
updated_date: '2026-05-05 05:20'
labels:
  - phase2.2
  - router-cleanup
  - issue-1116
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1292'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1116 Phase 2.2 by converting the minimal-test app optional vector and embedding router registrations for vector_stores_openai, embeddings_v5_production_enhanced, and media_embeddings from eager try/import RouterSpec blocks to ImportedRouterSpec-backed lazy router specs. Keep the tranche narrow and preserve existing prefixes, tags, route keys, and skip behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 minimal vector and embedding optional router specs defer module import and router attribute lookup until registration
- [x] #2 existing route metadata for vector-stores, embeddings, and media-embeddings is preserved
- [x] #3 focused router-group tests cover the lazy behavior with red/green verification
- [x] #4 router-group, main-router, and OpenAPI contract tests pass for the touched scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a narrow lazy-router tranche for minimal vector/embedding routers. Added red/green router-group coverage that proves vector_stores_openai, embeddings_v5_production_enhanced, and media_embeddings defer module import and router attribute access until ImportedRouterSpec resolution. Updated an adjacent older test expectation so those modules are no longer treated as unrelated eager imports.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted minimal-test optional vector/embedding router registration to ImportedRouterSpec while preserving existing prefixes, tags, route keys, and skip behavior. Verification: focused red/green test, targeted regression pair, full router_groups contract suite, main router contract suite, OpenAPI contracts, Bandit on minimal.py, git diff --check.
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
