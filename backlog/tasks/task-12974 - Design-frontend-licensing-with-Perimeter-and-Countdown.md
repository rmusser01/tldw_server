---
id: TASK-12974
title: Design frontend licensing with Perimeter and Countdown
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-20 19:08'
labels:
  - licensing
  - frontend
  - design
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-07-19-frontend-source-available-licensing-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the approved source-available licensing design for the WebUI, browser extension, shared UI package, and admin UI using unmodified PolyForm Perimeter 1.0.1 plus release-specific PolyForm Countdown 1.0.0, while retaining GPL-3.0-only for backend implementation and Apache-2.0 for the canonical OpenAPI contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The design selects unmodified PolyForm Perimeter 1.0.1 plus release-specific PolyForm Countdown grants that add AGPL-3.0-only on each release's second anniversary.
- [x] #2 Protected frontend paths, GPL-3.0-only backend implementation, and Apache-2.0 OpenAPI contract boundaries are explicit.
- [x] #3 The design preserves prior public grants and records draft PR #2727's historical boundary.
- [x] #4 Pre-counsel standard terms are separated from later counsel-reviewed community, customer, contributor, trademark, and commercial terms.
- [x] #5 Artifact isolation, notices, contribution intake, CI/release gates, failure handling, and accepted limitations are documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Consolidates the superseded BSL draft and final Perimeter/Countdown revision after rebasing onto latest dev exposed duplicate historical task IDs. The approved design spec was reviewed before implementation; Bandit is not applicable to Markdown-only design work.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed and reviewed the frontend licensing design. The approved model is standard PolyForm Perimeter 1.0.1 with rolling release-specific PolyForm Countdown 1.0.0 grants to AGPL-3.0-only, protected frontend path boundaries, unchanged GPL backend licensing, Apache OpenAPI licensing, preserved public history, and deferred counsel-reviewed custom grants.
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
