---
id: TASK-43
title: Phase 2.2 content ingestion router conditional cleanup R
status: Done
assignee: []
created_date: '2026-05-04 14:53'
updated_date: '2026-05-05 00:42'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
documentation:
  - >-
    Docs/superpowers/plans/2026-05-03-phase2-followup-stack-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue #1116 Phase 2.2 by deferring a medium conservative content ingestion/adapters tranche from iter_content_router_specs while preserving route metadata and optional-import behavior. Scope is limited to connectors, ingestion_sources, web_scraping, and reading_highlights after the audiobook router landed in PR #1271; kanban, notes/study/persona, and minimal router groups remain outside this tranche.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 connectors, ingestion_sources, web_scraping, and reading_highlights content router specs defer router attribute lookup until registration/resolution
- [x] #2 Existing prefix, tags, route_key, and default_stable behavior for the scoped content routers remain unchanged
- [x] #3 Focused red/green router laziness coverage, full router contract tests, main router/OpenAPI contracts, Bandit touched source scan, and git diff hygiene are run before commit
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Starting from stacked branch codex/phase2-2-content-ingestion-router-conditionals-r at local Q commit f58e634321; this branch will be rebased/PR'd after the earlier Phase 2.2 stack lands or kept stacked until then.

Verification: baseline full router group contract passed 58 before edits. Red focused ingestion/adapters laziness test failed before implementation because scoped router attrs were eagerly resolved. Green focused rerun passed 1 selected; full router group contract passed 59; main router contract passed 6; OpenAPI contract suite passed 69; Bandit content router group source reported 0 results and 0 errors; git diff --check passed.

After PR #1271 merged first, rebased this tranche onto origin/dev and removed the already-landed audiobook coverage from this task/test scope. The updated PR now covers connectors, ingestion_sources, web_scraping, and reading_highlights only.

Post-rebase verification: focused ingestion/adapters laziness test passed 1 selected; full router group contract passed 60; main router contract passed 6; OpenAPI contract suite passed 69; Bandit content router group source reported 0 results and 0 errors; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted connectors, ingestion_sources, web_scraping, and reading_highlights content router registrations to lazy ImportedRouterSpec entries while preserving prefixes, tags, route keys, and default_stable behavior. Added contract coverage proving iter_content_router_specs does not touch those router attributes during spec construction.
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
