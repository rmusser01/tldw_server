# CI Required Gates

This document defines the required pull-request gate contract during the phased CI rollout.

## Required Check Names

Configure branch protection to require these checks:

1. `backend-required`
2. `security-required`
3. `coverage-required`
4. `frontend-required`
5. `e2e-required`
6. `container-build-check` *(pending branch protection configuration)*

These check names are stable and should remain unchanged once branch protection is configured.

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

## Rollout Phases (2-4 Weeks)

1. Introduce required lanes and deterministic no-op semantics.
2. Tighten blocking behavior across required lanes.
3. Refine path coupling and flake handling in `e2e-required`.
4. Finalize branch protection to required lane names above.
5. Add `container-build-check` to branch protection required statuses.

## Legacy CI Workflow

The large legacy `.github/workflows/ci.yml` workflow remains available during rollout for broad visibility and historical comparison.
Required merge protection is provided by the six lanes listed above.
