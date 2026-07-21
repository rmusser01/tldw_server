---
id: TASK-12977
title: Bootstrap base-controlled frontend license gate
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-21 00:15'
labels:
  - licensing
  - security
  - ci
  - frontend
dependencies: []
references:
  - >-
    https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows
  - 'https://docs.github.com/en/actions/reference/security/secure-use'
  - >-
    https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets
documentation:
  - >-
    Docs/superpowers/specs/2026-07-20-base-controlled-frontend-license-gate-design.md
  - >-
    Docs/superpowers/plans/2026-07-20-base-controlled-frontend-license-gate-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the PR-controlled temporary frontend contribution gate with a base-controlled GitHub Actions trust root. Design and implement a read-only pull_request_target workflow on the default branch, NUL-safe changed-path classification, and required-status rulesets for dev and main, while ensuring the workflow never executes or checks out untrusted pull-request code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A reviewed design documents why the pull_request gate is insufficient and defines the default-branch trust root, two-PR rollout, ruleset bootstrap, and rollback behavior.
- [ ] #2 The gate workflow runs from the default branch with contents: read, never checks out or executes pull-request code, and evaluates immutable pull-request identity and changed paths.
- [x] #3 Changed paths are transported and parsed NUL-safely with adversarial filename coverage and exact protected/governance/API boundaries.
- [ ] #4 The bootstrap workflow lands on main before the dev licensing cutoff and the gate status becomes required on both dev and main without weakening existing rules.
- [ ] #5 Owner-authored cutoff changes pass while external protected or governance changes fail, and all workflow, security, and focused tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review of Task 4 in TASK-12976 proved that a pull_request workflow is PR-controlled and that line-oriented changed paths are not exact. Live GitHub inspection found dev unprotected, ruleset 5653432 active only on the default branch without required checks, and rmusser01 as the only direct collaborator. The approved replacement uses a default-branch pull_request_target workflow, NUL-safe metadata-only diffing, branch-specific trusted commit-status contexts for main and dev, and staged ruleset activation. Robert Benjamin Jake Musser approved the written design on 2026-07-20. The implementation plan stages a main bootstrap, live expected-source verification, guarded ruleset activation, and reconciliation back into TASK-12976. External ruleset mutation remains a post-bootstrap step.

- Task 1 created the isolated bootstrap worktree at `.worktrees/frontend-license-gate-bootstrap` from `origin/main` `7a23be3202`; commit `e66028e959` recorded the bootstrap and the baseline workflow test passed 1/1.
- Task 2 landed in `6dd2d64231`: RED was the expected missing classifier module, GREEN was 16/16, Ruff, Black, Bandit, and diff checks were clean, and independent review found no issues.
- Task 3 landed across `6a681aacde`, `405ccc6b4b`, `bdec6c1777`, and `e16927b3eb`; mutation-driven contract hardening closed the privileged execution surface, the final focused set passed 27/27, actionlint 1.7.12, Ruff, Black, Bandit, and diff checks were clean, and independent task quality was Approved.
- Broad final review found that one `frontend-license-policy/trusted` context was unsafe across `main` and `dev`: GitHub keys commit statuses by repository, head SHA, and context, so a success computed for one base could satisfy the other's loose required-status rule for the same head SHA. Robert approved branch-specific `/main` and `/dev` contexts, the `edited` trigger, exact privileged job/concurrency assertions, and consistent design/plan/ruleset/rollback corrections on 2026-07-20.
- Pre-fix evidence showed the existing workflow contract passing 10/10 while both supported bases selected the same shared context. The new regression then failed as expected with 4 failures and 8 passes: the workflow lacked `edited` and still exposed the shared literal context.
- Commit `e1b84cbcf0` derives `STATUS_CONTEXT` from GitHub's base ref, adds `edited`, rejects a shared context, proves `/main` and `/dev` are distinct, and locks the exact workflow, concurrency, job, runner, timeout, environment, and step surfaces. Focused GREEN was 12/12; the combined classifier/workflow/frontend-required set was 29/29.
- Commit `ae71b9310b` synchronizes the approved design and plan: main ruleset `5653432` requires only `/main`, the dev ruleset requires only `/dev`, licensing-PR observation queries `/dev`, rollback/evidence covers both contexts, and the classifier inventory correctly says six protected prefixes. Plan Stages 1-3 remain Complete; Stages 4-5 remain Not Started.
- Final verification passed 29/29 focused tests with 6 pre-existing warnings; controller-owned actionlint 1.7.12 exited 0 with no findings for `frontend-license-gate.yml` and `actionlint.yml`; Ruff and Black were clean; Bandit reported 0 errors and 0 findings; marker counts were one for each notes/final-summary boundary; the stale shared-context scan and `git diff --check` were clean. AC #3 remains checked while live AC #2, #4, and #5 remain open.
- Backlog MCP/CLI was unavailable despite repeated attempts. Robert approved this narrowly scoped direct-edit exception on 2026-07-20, including the marker-preserving final-review update.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
