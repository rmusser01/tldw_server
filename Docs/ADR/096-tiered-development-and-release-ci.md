# ADR-096: Tier server development and release CI

**Status:** Proposed
**Date:** 2026-11-18
**Decision owner:** Repository maintainer
**Related task:** TASK-15572
**Related plan:** [Server CI tiers](https://github.com/rmusser01/tldw_server/blob/dev/Docs/Development/CI_REQUIRED_GATES.md#server-ci-tiers)

## Decision

Use focused CI for PRs targeting `dev`, including PRs from forks. Keep the existing required checks and license admission. In the supplemental workflow, the existing path classifier selects bounded backend smoke when backend paths change. Other `dev` PRs run the regular preflight checks and skip backend smoke.

Run the existing broad Linux matrix against the dispatched `dev` commit nightly. Skip the dispatch if that commit already has a successful or active nightly. A failed attempt may be retried. Existing specialized workflows keep their schedules and platform coverage.

A PR into `main` runs the complete existing server matrix and validates the built distributions on GitHub's PR merge commit. Require the `Main PR release qualification` check at the current base. Rolling images, tagged images and PyPI uploads require that same qualification in their run at the event commit. Keep the top-level PyPI workflow and its environments for Trusted Publishing. Run the rolling image and eligible automatic PyPI upload from one qualified `main` push. Standalone MCP Unified has its own pipeline.

## Context

The server already has focused required checks and a larger supplemental matrix. The latter gives relevant backend PRs hundreds of Linux jobs while Windows and macOS mostly run on main pushes, release and manual CI. Package and image publication have separate triggers and can run before the full matrix reports.

## Alternatives considered

- Limit smoke CI to a path allowlist. Most `dev` changes would still launch hundreds of supplemental jobs and compete with the merge queue for runners.
- Look up another workflow's successful status before publish. Reusable workflows keep the test and publish dependencies in the same run and avoid associating another commit or a skipped check with release qualification.

## Consequences

The existing required `dev` workflows continue to exercise affected backend, security, coverage, frontend, and browser checks. Nightly preserves broader Linux coverage. Regressions covered only by the full matrix can surface during nightly or `main` qualification. Releases wait for all expected tests and package validation.

The focused `dev` route takes effect with the workflow change. Verify the `main` PR check on the CI-only promotion, then enforce its GitHub Actions context alongside the existing license status. The promotion installs the default-branch nightly schedule. See [CI required gates](https://github.com/rmusser01/tldw_server/blob/dev/Docs/Development/CI_REQUIRED_GATES.md#server-ci-tiers) for rollout checks.
