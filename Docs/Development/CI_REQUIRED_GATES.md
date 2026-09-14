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

Each required gate always reports a status for deterministic branch protection behavior.

- If relevant paths changed, the gate executes its full checks.
- If relevant paths did not change, the gate exits with an explicit no-op success message.

Examples:

- UI-only PRs no-op `backend-required` and `coverage-required`.
- Backend-only PRs no-op `frontend-required`.
- `e2e-required` runs on frontend changes and selected backend API/schema/auth paths.

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
