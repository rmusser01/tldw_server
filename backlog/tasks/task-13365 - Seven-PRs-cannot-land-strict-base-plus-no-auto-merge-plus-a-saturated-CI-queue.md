---
id: TASK-13365
title: >-
  Seven PRs cannot land: strict base plus no auto-merge plus a saturated CI
  queue
status: To Do
assignee: []
created_date: '2026-09-23 18:25'
labels:
  - ci
  - process
  - blocked
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
This is a structural deadlock in the merge path, not a property of any one PR. Four facts combine, all measured on 2026-09-23:

1. **The ruleset requires a current base.** `dev-core-required-gates` (id 21824526) has `strict_required_status_checks_policy: true` over six checks: backend-required, security-required, coverage-required, frontend-required, e2e-required, container-build-check. So a PR must be up to date with `dev` at the moment of merge.
2. **Auto-merge is off.** `GET /repos/rmusser01/tldw_server` returns `allow_auto_merge: false` and `allow_update_branch: false`. So the merge cannot be queued to fire when gates pass, and the lightweight "Update branch" is unavailable -- the only way to become current is a rebase push, which invalidates every check and restarts the full cycle.
3. **No bypass exists.** `bypass_actors: []`, `current_user_can_bypass: "never"`, `enforcement: active`.
4. **Cycle time exceeds the interval between `dev` advances.** `dev` merged at 00:02, 03:02, 05:58, 06:28, 09:26 and 11:02 local -- gaps of 24 to 180 minutes, median around 90. A full check cycle under the current queue is taking over two hours: PR #2996 was pushed at 14:43 UTC and still had 10 checks pending more than two hours later.

**Consequence.** Rebasing to satisfy (1) restarts a cycle that takes longer than the next `dev` advance, so the PR is BEHIND again before it finishes. Observed directly: #2992 and #2996 were both rebased onto `183e8ea`, and `dev` moved to `158287d` (#2999) before either completed. Seven open PRs, none landed.

The amplifier is TASK-13359: every required gate currently runs **twice** per PR, so cycle time is roughly double what it needs to be.

**Levers, all owner-side. Any one of the first two breaks the deadlock:**

- **Enable `allow_auto_merge`.** Then each PR merges the instant its gates pass, and the base-update race disappears because GitHub sequences it. This is the smallest change with the largest effect.
- **Relax `strict_required_status_checks_policy`.** Keeps the six required checks but stops requiring the branch to be current. Weakens the guarantee that checks ran against the merged state -- a real tradeoff, not free.
- **Fix TASK-13359** so gates run once. Halves cycle time; may not be sufficient alone, since a one-hour cycle still loses to a 24-minute gap.
- **Enable `allow_update_branch`** so becoming current is a merge commit rather than a rebase push. Still re-runs CI, so this helps least.

I cannot change any of these -- they are repository settings.

Affected: #2992, #2994, #2996, #2997, #3000, #3004, #3006, none of which has a failing required gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A PR with all six required gates green can reach merged state without manually winning a race against dev
- [ ] #2 The chosen lever is recorded, including the tradeoff if strict mode is relaxed
- [ ] #3 Measured: cycle time and dev-advance interval after the change
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
