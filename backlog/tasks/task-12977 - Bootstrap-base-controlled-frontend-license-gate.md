---
id: TASK-12977
title: Bootstrap base-controlled frontend license gate
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-21 03:59'
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
- [x] #2 The gate workflow runs from the default branch with contents: read, never checks out or executes pull-request code, and evaluates immutable pull-request identity and changed paths.
- [x] #3 Changed paths are transported and parsed NUL-safely with adversarial filename coverage and exact protected/governance/API boundaries.
- [x] #4 The bootstrap workflow lands on main before the dev licensing cutoff and the gate status becomes required on both dev and main without weakening existing rules.
- [x] #5 Owner-authored cutoff changes pass while external protected or governance changes fail, and all workflow, security, and focused tests pass.
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
- Final full-branch review then clarified that a required commit status is identified by repository, head SHA, and branch-specific context rather than by pull-request author. Robert approved the exact-commit authorization model on 2026-07-20: an owner-authored run authorizes that immutable SHA for the selected base; another PR may use the authorization only with the identical SHA/base pair; and a third-party rerun can fail the shared identity only as denial of service, not authorize different code.
- The approved rollout correction keeps plan Stages 1-3 Complete and Stages 4-5 Not Started. After Robert supplies his own bootstrap PR Change summary and the bootstrap merges, Stage 4 first opens a harmless owner-controlled empty-commit validation PR to `main`, records a real source-bound `/main` result, closes the draft, and removes the temporary branch. It separately records the licensing PR's `/dev` result before either matching required-status rule is activated; a missing or mismatched expected source stops the rollout without an any-source fallback.
- Exact-commit wording was implemented test-first: the focused workflow contract failed 2/12 against the old change-scoped descriptions, then passed 12/12 after the two status descriptions and their pinned script digests were updated. The status publisher still targets `${{ github.event.pull_request.head.sha }}` and the branch-qualified base context; permissions, triggers, checkout, and execution surfaces are unchanged. Live AC #2, #4, and #5 remain open.
- Post-correction verification passed the full 29/29 licensing-gate matrix with 6 pre-existing warnings; actionlint 1.7.12 reported no findings for both targeted workflow files; Ruff and Black were clean for the changed test. No production Python changed, so a new Bandit run was not applicable. Marker counts remained 1/1/1/1, and stale wording plus correction-range/full-bootstrap `git diff --check` scans were clean.
- Backlog MCP/CLI was unavailable despite repeated attempts. Robert approved this narrowly scoped direct-edit exception on 2026-07-20, including the marker-preserving final-review update.
- Technical rollout completed on 2026-07-21. Bootstrap PR #2753 merged as `d9c245ac14c40df855d1ab6cd19b3c137b16b47b`; however, its required human-written `Change summary` remained empty. That repository-policy requirement was not satisfied and is recorded as known noncompliance rather than retroactively claimed complete.
- Temporary owner validation PR #2754 proved `frontend-license-policy/trusted/main` on head `2499d9f09f514bc29e3455cc7b640408a8a2f510` in successful workflow run `29802687578` / job `88546692070`, then closed unmerged; its remote branch, local branch, and dedicated worktree were removed.
- Draft licensing PR #2755 proved `frontend-license-policy/trusted/dev` on head `064ab2569632f55ad057fd1c02bc0c94709e1d18` in successful workflow run `29810555026` / job `88570361051`. Both successful audit checks were created by `github-actions[bot]` through GitHub Actions App integration `15368` and workflow ID `317148516`.
- Main ruleset `5653432` remains active with its exact prior conditions, no bypass actors, deletion rule, non-fast-forward rule, and pull-request rule; the sole addition is source-bound `frontend-license-policy/trusted/main` with `strict=false`. Active dev ruleset `19362594` targets only `refs/heads/dev`, has no bypass actors, copies the pull-request rule, and requires only source-bound `frontend-license-policy/trusted/dev` with `strict=false`. Effective-rules readback confirmed both contexts, and PR #2727 is blocked because its head lacks the required `/dev` status.
- Public ruleset snapshots are recorded under `Docs/superpowers/evidence/TASK-12977/`. Main was activated at `2026-07-21T00:43:19.243-07:00`; dev was activated at `2026-07-21T00:43:49.962-07:00`.
- Stage 5 reconciliation replaced the rejected PR-controlled step in `frontend-required.yml` with a negative regression contract and carried the reviewed trusted workflow, classifier, tests, and actionlint target from merged `main`. RED failed because the rejected workflow lacked the conditional checkout; GREEN passed 2/2 after restoring the workflow from `origin/dev`. Fresh final verification passed 40/40 focused tests with six pre-existing warnings; pinned actionlint 1.7.12, Ruff, Black, Bandit with zero findings/errors across 74 classifier LOC, deterministic owner/external cases, evidence assertions, marker integrity, and `git diff --check` all passed. Independent code/security review was CLEAN on the base-control and NUL/rename findings. Its documentation finding was resolved by replacing the stale rejected Task 4 instructions, and the plan re-review was CLEAN.
- Reconciled commit `f7c635d34749663fcb52a5ee93561d8013bad022` passed source-bound `frontend-license-policy/trusted/dev`. Replacement run `29813192487` / job `88578513698` completed successfully through workflow `317148516` and GitHub Actions App `15368`; PR #2755 remained draft and PR #2727 remained held behind the cutoff.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and activated the base-controlled frontend license gate. The trusted workflow is on `main`; branch-specific source-bound rules protect `main` and `dev`; public ruleset evidence is committed; and the licensing branch now uses the reviewed NUL-safe classifier while `frontend-required.yml` has no licensing authority. Local verification and independent security review are clean, and the reconciled PR #2755 head passed the required `/dev` gate. Bootstrap PR #2753's missing human-written Change summary remains documented repository-policy noncompliance.
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
