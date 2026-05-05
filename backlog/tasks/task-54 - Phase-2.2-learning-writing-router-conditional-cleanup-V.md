---
id: TASK-54
title: Phase 2.2 learning/writing router conditional cleanup V
status: Done
assignee:
  - codex
created_date: '2026-05-05 02:11'
updated_date: '2026-05-05 02:15'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1279'
  - 'https://github.com/rmusser01/tldw_server/pull/1280'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue #1116 Phase 2.2 with a content-tail learning/writing tranche. Scope is limited to translate/slides/flashcards/quizzes/study_suggestions/writing/writing_manuscripts eager router imports in tldw_Server_API/app/api/v1/router_groups/content.py; open notes/minimal PRs remain separate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Content translate/slides/flashcards/quizzes/study/writing router specs defer module import and router attr lookup until registration or resolution
- [x] #2 Existing prefixes/tags/route keys/default stability for those content routers remain unchanged
- [x] #3 Focused red/green coverage, full router group tests, main router/OpenAPI contracts, Bandit touched source scan, and git diff hygiene are run before commit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused content-router contract coverage proving translate/slides/learning/writing modules are not imported and router attrs are not touched during iter_content_router_specs construction. 2. Run the focused test red on origin/dev behavior. 3. Replace only the selected content.py eager try blocks with ImportedRouterSpec entries. 4. Re-run focused/full router groups, main router contracts, OpenAPI contracts, Bandit on content.py, and diff hygiene. 5. Commit, push, create PR against dev, and update #1116.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: focused red failed on origin/dev with eager attr lookup for translate/slides/flashcards/quizzes/study_suggestions/writing/writing_manuscripts; focused green passed after patch (1 passed, 61 deselected); full router groups passed (62 passed); main router contract passed (6 passed); OpenAPI contracts passed (69 passed); Bandit on content.py reported 0 results and 0 errors; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted only the content-tail translate/slides/learning/writing eager try blocks to ImportedRouterSpec entries, preserving prefixes/tags/route keys/default stability while deferring imports and router attr lookup until registration or resolution.
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
