# CI Required Gates

This document defines the required pull-request gate contract for `dev`.

## Required Check Names

The active `dev-core-required-gates` repository ruleset requires these checks:

1. `backend-required`
2. `security-required`
3. `coverage-required`
4. `frontend-required`
5. `e2e-required`
6. `container-build-check`

These check names are stable and must remain unchanged. Each status is bound to
GitHub Actions integration `15368`.

## Live `dev` Enforcement

GitHub aggregates two active, `dev`-only repository rulesets:

- `dev-core-required-gates` (ID `21824526`) requires all six checks above,
  uses strict/current-base enforcement, and has no bypass actors.
- `frontend-license-gate-dev` (ID `19362594`) retains the pull-request rule
  and requires `frontend-license-policy/trusted/dev` from integration `15368`.

The effective pull-request rule sets the required approving-review count to
zero, dismisses stale reviews after a push, and requires extra approval for
unattributed changes.
Neither ruleset grants an administrative bypass; GitHub reports
`current_user_can_bypass: never` for both.

If the additive core ruleset malfunctions, disable ruleset `21824526` without
deleting it or modifying ruleset `19362594`, then record the before/after API
responses in TASK-13013.2.

### Container Build Check Details

`container-build-check` validates that the `app`, `webui`, and `admin-ui` Dockerfiles build successfully on PRs to `main` and `dev`. The workflow uses a matrix strategy with `fail-fast: false`, so all three images are tested even if one fails. A summary job rolls up the matrix results into a single `container-build-check` status for branch protection.

See [Container Image Lifecycle](Container_Image_Lifecycle.md) for the full build and publish pipeline.

## Conditional Execution and No-op Behavior

Each required gate is *designed* to always report a status, so branch protection behaves
deterministically. It currently does not -- see
[Known Defect: Gates Are Cancelled Before They Report](#known-defect-gates-are-cancelled-before-they-report).

- If relevant paths changed, the gate executes its full checks.
- If relevant paths did not change, the gate exits with an explicit no-op success message.

Examples:

- UI-only PRs no-op `backend-required` and `coverage-required`.
- Backend-only PRs no-op `frontend-required`.
- `e2e-required` runs on frontend changes and selected backend API/schema/auth paths.

## Known Defect: Gates Are Cancelled Before They Report

The section above says each required gate always reports a status. **In practice it
usually does not**, and a pull request stays `BLOCKED` indefinitely until someone
manually re-runs the gates. Measured 2026-09-22 across four PRs (#2981-#2984) and six
pushes: every check went to `CANCELLED` within roughly 40 seconds of each push -- 45 to
50 per PR, all six required gates included, with zero `FAILURE`. See TASK-13355.

### Mechanism

1. A pull-request event fires `Frontend License Gate Audit`, which triggers on
   `pull_request_target` with types `[opened, reopened, synchronize, ready_for_review,
   edited]`. **`edited` is included, so editing a PR description counts.**
2. When that audit completes it fires every workflow declaring
   `workflow_run: [Frontend License Gate Audit]`. Those runs are attributed to the
   default branch, so they are invisible when listing runs for the PR branch.
3. A `workflow_run` run resolves its concurrency group through
   `github.event.workflow_run.pull_requests[0].number` -- the same PR number the
   `pull_request` run used -- so both share one group and `cancel-in-progress: true`
   kills the `pull_request` run.
4. That `workflow_run` run's own `admission` job requires
   `vars.LICENSE_FIRST_CI_ENABLED == 'true'`. No repository variables are currently set,
   so it is `SKIPPED`.
5. Each gate job requires `needs.admission.result == 'success'` for `workflow_run`
   events, so it skips too. Directly observed: `event=workflow_run` runs of
   `backend-required`, `coverage-required`, `frontend-required` and `pre-commit` all at
   `completed/skipped`.

The run that would have reported a status is cancelled by a run that then reports
nothing.

### Landing a pull request today

Until TASK-13355 is resolved, this sequence works and nothing else reliably does:

1. **Rebase onto the current `dev` and push.** `dev-core-required-gates` uses
   strict/current-base enforcement, so a branch even one commit behind cannot merge, and
   GitHub reports the required checks as "expected" rather than counting the ones that
   already passed on the older base.
2. **Wait for the audit on that exact head SHA to complete.** Not merely "the latest
   audit" -- a rebase-push spawns a new one, and re-running gates while it is pending
   gets them cancelled when it finishes.
3. **Re-run the six gates individually, paced roughly 40 seconds apart.** Re-running
   them in a burst makes them cancel each other; re-running one that is already queued
   or in progress fails and silently skips it.
4. **Merge once all six are green**, before anything else lands on `dev`.

Two traps worth stating outright:

- **Do not edit the PR description after opening it** if you want its gates to survive.
  `pull_request_target` fires on `edited`, which spawns an audit and cancels every gate
  run.
- **`--admin` does not help.** `dev-core-required-gates` has no bypass actors, so
  `gh pr merge --admin` is refused with "Repository rule violations found / 6 of 6
  required status checks are expected". Merging through the GitHub UI as a user whose
  own admin bypass applies does work.

Note also that `--squash` is rejected repository-wide; merges must use `--merge`.

## Security Threshold Policy

`security-required` enforces blocking findings at `HIGH`/`CRITICAL` severity with an allowlist.

- Allowlist file: `.github/security/ci-allowlist.yml`
- Every allowlist entry must include:
  - vulnerability id
  - owner
  - expiry date (ISO format)

Expired allowlist entries are ignored by the gate.

## Rollout Status

1. Introduce required lanes and deterministic no-op semantics.
2. Tighten blocking behavior across required lanes.
3. Refine path coupling and flake handling in `e2e-required`.
4. Enforce the required lane names on `dev` with strict/current-base checks.
5. Enforce `container-build-check` with the other required statuses.

The required-status enforcement described in phases 4 and 5 is complete for
`dev` as of 2026-08-29. The controlled TASK-13013.2 proof PR recorded a
failing `coverage-required` check from integration `15368` and GitHub reported
the ready pull request as `BLOCKED`.

## Legacy CI Workflow

The large legacy `.github/workflows/ci.yml` workflow remains available during rollout for broad visibility and historical comparison.
Required merge protection is provided by the six lanes listed above.
