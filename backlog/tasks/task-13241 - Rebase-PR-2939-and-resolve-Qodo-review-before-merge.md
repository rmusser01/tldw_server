---
id: TASK-13241
title: Rebase PR 2939 and resolve Qodo review before merge
status: In Progress
assignee: []
created_date: '2026-09-10 06:30'
updated_date: '2026-09-10 06:31'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2939'
documentation:
  - Docs/Plans/IMPLEMENTATION_PLAN_pr2939_rebase_qodo_merge.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User requests rebasing PR2939 onto latest dev, addressing all Qodo issues/comments and subsequent review feedback, then merging. Preserve original fixes and UAT evidence. Verify latest reviewed head and repository merge requirements, including the human-owned Change summary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PR branch is rebased onto current dev with original regression fixes preserved
- [ ] #2 Every Qodo finding has a verified fix or evidence-based disposition and regression coverage where behavior changes
- [ ] #3 Fresh affected tests, lint, Bandit, review, and required CI checks pass on the final head
- [ ] #4 Merge into dev only after review requirements and repository human-authored Change summary are satisfied
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Track work, fetch dev/reviews, preserve branch, rebase and inspect changes. 2. Reproduce and fix Qodo correctness findings; improve tests/docs and resolve CI failures. 3. Verify final changes, publish responses, wait for fresh review/checks, and merge once requirements are satisfied.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Initial remote PR head5065421d801490bd45a35d355b7eaa6f00e44b7f; latest fetched dev9da94ebcb41ba45d279ab707ec92a285d31ba5c7,17 commits ahead of prior base. Qodo posted seven findings on current PR head: queued macro default snapshot, silent config fallback, migration/test docstrings, test annotations, private array mechanics, and DB-layer ownership. Current CI has a shard-coverage guard failure. Independent bounded investigations assigned for CI, architecture, and test quality while coordinator handles rebase and macro/config fixes.
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
