---
id: TASK-12982
title: 'Land PR #2727 on current dev'
status: In Progress
assignee: []
created_date: '2026-07-22 03:45'
updated_date: '2026-07-22 05:30'
labels:
  - integration
  - release
  - licensing
  - ci
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2727'
  - TASK-12963
documentation:
  - Docs/superpowers/specs/2026-07-21-pr-2727-landing-private-pilot-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate the merged frontend licensing cutoff into PR #2727, revalidate the exact head, satisfy review and human-authorship requirements, and merge the provider credential runtime into dev without disturbing the user-owned dirty worktree.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Current dev is integrated into PR #2727 without losing its reviewed feature commits or the pre-existing user-owned worktree changes.
- [ ] #2 Fresh exact-head required CI and frontend-license trusted checks pass, with reproduced failures fixed rather than bypassed.
- [ ] #3 The requester supplies the required human-written Change summary, PR #2727 is marked ready, and it merges into dev.
- [ ] #4 Landing evidence records integration parents, exact-head gates, reproduced failure dispositions, review results, and the final merge commit.
- [ ] #5 The actual merge commit is verified to contain the validated PR head and current protected dev tip, with merged licensing metadata and trusted-policy files present, before any deployment task begins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-21-pr-2727-current-dev-landing-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-07-21: During design finalization, a separate owner process committed the previously dirty follow-up as 7d76bdfcc0, merged protected dev 8ed612c7e0 in conflict-free merge 0e8eadc55f (first parent 7d76bdfcc0, second parent 8ed612c7e0), recorded post-merge validation in 6065c64ab4, and pushed that exact PR head. Both original PR head e8bcc4c8b and protected dev are ancestors. Fresh exact-head CI is in progress; no non-green context is waived.

Execution baseline reconfirmed: PR head 6065c64ab4 contains original head e8bcc4c8b and protected dev 8ed612c7e0 through merge 0e8eadc55f (parents 7d76bdfcc0 and 8ed612c7e0). Local work descends from that head. The index is empty; protected concurrent out-of-scope dirty paths are the tracked TASK-12963 and TASK-2234 Backlog records, plus untracked server-ux-smoke.pid and the two named watchlist templates; all remain unstaged and excluded from this task. PR #2727 remains draft and mergeable/unstable at base 8ed612c7e0; frontend-license-policy/trusted/dev passes. Current non-passing checks are actionlint and backend-required failures, a Windows Python 3.12 Full Suite failure, and one canceled Windows research-websearch shard. Known current-head corrections are actionlint SC2155 and OpenAPI fingerprint drift.
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
