---
id: TASK-13112
title: Rebase PR 2808 recipient shared data plane onto latest dev
status: In Progress
assignee: []
created_date: '2026-08-23 04:10'
labels:
  - workspaces
  - sharing
  - rebase
  - frontend
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2808'
  - TASK-12020.40
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase codex/research-workspace-power-user-uat onto the latest origin/dev, resolve integration conflicts without regressing current dev or the canonical recipient shared-workspace data plane, verify the affected backend/frontend contracts, and update PR #2808.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Branch history is rebased onto the latest fetched origin/dev.
- [ ] #2 All conflicts preserve current dev behavior and the canonical recipient shared-workspace contract.
- [ ] #3 Affected frontend/backend tests and repository integrity checks pass or documented unrelated baselines remain.
- [ ] #4 PR #2808 is updated with force-with-lease and its merge state is verified.
- [ ] #5 The unrelated untracked watchlist templates remain untouched and excluded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-08-23-pr-2808-dev-rebase.md
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
