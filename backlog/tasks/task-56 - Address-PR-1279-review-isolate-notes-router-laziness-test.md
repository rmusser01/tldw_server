---
id: TASK-56
title: 'Address PR #1279 review: isolate notes router laziness test'
status: Done
assignee:
  - codex
created_date: '2026-05-05 02:38'
updated_date: '2026-05-05 02:43'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1279'
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve Qodo PR #1279 reliability finding that the notes laziness contract test resolves every content RouterSpec instead of only the targeted notes_graph, notes, and web_clipper specs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Notes laziness test filters selected specs before resolving router factories
- [x] #2 Test still verifies notes_graph, notes, and web_clipper metadata plus lazy attr lookup behavior
- [x] #3 Focused/full router group, main router/OpenAPI contracts, Bandit or documented test-only skip, and diff hygiene are run before commit
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Addressed PR #1279 Qodo reliability finding after rebasing the branch onto origin/dev. The notes laziness test now selects only notes_graph, notes, and web_clipper specs by ImportedRouterSpec log_name before resolving router factories, so unrelated content routers are not imported by this focused test.

Added importlib tracking for the three targeted notes modules. list(iter_content_router_specs()) leaves import_calls and router attr access at zero; resolving the selected specs imports and resolves only those three routers once.

Verification passed post-rebase: focused notes laziness test; full router groups; main router contract; OpenAPI contracts; Bandit content.py 0 results and 0 errors; git diff --check.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR #1279 review feedback by isolating the notes laziness test to the targeted notes_graph, notes, and web_clipper specs before router factory resolution. Rebased the PR branch onto latest origin/dev and verified focused/full router contracts, main router contracts, OpenAPI contracts, Bandit, and diff hygiene.
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
