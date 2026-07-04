---
id: TASK-12141
title: Address PR 2598 review feedback
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-04 17:57'
labels:
  - mcp
  - docs
  - review
dependencies: []
documentation:
  - 'https://github.com/rmusser01/tldw_server/pull/2598'
  - >-
    Docs/superpowers/plans/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR 2598 on latest dev and address actionable review feedback for standalone MCP docs source discovery. Scope is limited to review items on sitemap/XML robustness, URL origin handling, source registration lifecycle, sync item reason reporting, small docstring/type/style comments, tests, Bandit, and PR push.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PR branch is rebased on latest origin/dev and pushed.
- [ ] #2 Actionable PR review issues are fixed in code/tests or explicitly documented as not applicable.
- [ ] #3 Focused tests cover XML parse failure propagation, default-port origin equivalence, query sitemap registration denial, and sync skipped reason propagation.
- [ ] #4 MCP docs tests, import-boundary tests, Bandit, and diff hygiene pass.
- [ ] #5 Backlog task records final verification and PR status.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use the existing PR branch. Rebase on origin/dev, inspect review threads, implement the smallest root-cause fixes with focused tests, run MCP docs tests plus Bandit, update this Backlog task, and force-push the rebased PR branch.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Bandit run for touched code when applicable or documented skip
- [ ] #4 Final summary added
- [ ] #5 Known skips or blockers documented
<!-- DOD:END -->
