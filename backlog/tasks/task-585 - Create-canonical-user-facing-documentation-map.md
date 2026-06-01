---
id: TASK-585
title: Create canonical user-facing documentation map
status: In Progress
assignee: []
created_date: '2026-06-01 05:24'
updated_date: '2026-06-01 05:25'
labels: []
dependencies: []
documentation:
  - Docs/User_Guides/index.md
  - Docs/mkdocs.yml
  - README.md
  - apps/extension/docs/index.md
  - Docs/superpowers/specs/2026-06-01-user-docs-map-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement the first documentation IA slice: a canonical public user docs hub under Docs/User_Guides that improves discoverability across server API, WebUI, and browser extension docs without moving deep pages in the first pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Approved design spec exists for the canonical user documentation map.
- [ ] #2 Implementation plan covers the hub rewrite, optional feature map, MkDocs nav, README pointer, and extension docs pointer.
- [ ] #3 Backlog task records touched files and verification results before closeout.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved brainstorming design written to Docs/superpowers/specs/2026-06-01-user-docs-map-design.md. Scope is canonical Docs/User_Guides hub, optional feature map, MkDocs nav, README pointer, and extension docs pointer; WebUI documentation behavior is deferred.
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
