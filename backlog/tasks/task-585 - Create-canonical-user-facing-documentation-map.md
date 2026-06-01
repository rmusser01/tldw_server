---
id: TASK-585
title: Create canonical user-facing documentation map
status: In Progress
assignee: []
created_date: '2026-06-01 05:24'
updated_date: '2026-06-01 19:25'
labels: []
dependencies: []
documentation:
  - Docs/User_Guides/index.md
  - Docs/User_Guides/Feature_Map.md
  - Docs/mkdocs.yml
  - README.md
  - apps/extension/docs/index.md
  - Docs/superpowers/specs/2026-06-01-user-docs-map-design.md
  - Docs/superpowers/plans/2026-06-01-user-docs-map.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement the first documentation IA slice: a canonical public user docs hub under Docs/User_Guides that improves discoverability across server API, WebUI, and browser extension docs without moving deep pages in the first pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Approved design spec exists for the canonical user documentation map.
- [x] #2 Implementation plan covers the hub rewrite, optional feature map, MkDocs nav, README pointer, and extension docs pointer.
- [ ] #3 Backlog task records touched files and verification results before closeout.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved brainstorming design written to Docs/superpowers/specs/2026-06-01-user-docs-map-design.md. Scope is canonical Docs/User_Guides hub, optional feature map, MkDocs nav, README pointer, and extension docs pointer; WebUI documentation behavior is deferred.

Design review before implementation planning tightened the spec around public link targets, MkDocs Home behavior, exact MkDocs build command, and generated Docs/Published commit policy.

Implementation plan written to Docs/superpowers/plans/2026-06-01-user-docs-map.md. The plan creates a separate Feature_Map.md, rewrites Docs/User_Guides/index.md as the hub, updates MkDocs/README/extension entry points, refreshes Docs/Published, and records docs-only verification.

Task 4 verification: refreshed curated docs with Helper_Scripts/refresh_docs_published.sh. Kept generated Docs/Published/User_Guides/index.md, Docs/Published/User_Guides/Feature_Map.md, and Docs/Published/Getting_Started/TROUBLESHOOTING.md because the generated user guide hub links to the troubleshooting guide. Changed Markdown link check passed. git diff --check passed. MkDocs build was attempted with the project virtualenv from the main worktree and failed because mkdocs is not installed: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python: No module named mkdocs. Bandit is not applicable for this docs-only slice because no Python or executable code was changed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
