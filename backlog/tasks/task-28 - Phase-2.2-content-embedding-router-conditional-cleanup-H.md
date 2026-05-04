---
id: TASK-28
title: Phase 2.2 content embedding router conditional cleanup H
status: Done
assignee: []
created_date: '2026-05-04 02:52'
updated_date: '2026-05-04 02:54'
labels:
  - phase-2.2
  - router-groups
  - tests
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1250'
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After #1251 merged the chunking/vector/prompts processing tranche, keep PR #1250 focused on the remaining content embedding routers. Defer router attribute lookup for embeddings_v5_production_enhanced and media_embeddings while preserving route metadata and ordering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Content embeddings and media embeddings router specs defer router attribute lookup until resolution.
- [x] #2 Existing prefixes, tags, route keys, and router ordering are preserved.
- [x] #3 Focused and full router-group verification pass after rebase.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
After #1251 merged the processing router tranche, rebased PR #1250 and narrowed this slice to embeddings_v5_production_enhanced and media_embeddings. Preserved #1251 chunking/vector/prompts tests and added separate embedding router attr laziness coverage.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Deferred the remaining content embedding router imports behind ImportedRouterSpec factories while preserving prefix/tag/route_key metadata. Verification after rebase: focused router-group selection, full router_groups contract, main_router contract, OpenAPI contracts, Bandit on conditional.py/content.py, and git diff --check.
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
