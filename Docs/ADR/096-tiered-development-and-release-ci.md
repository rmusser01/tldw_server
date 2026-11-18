# ADR-096: Tier server development and release CI

**Status:** Proposed
**Date:** 2026-11-18
**Decision owner:** Repository maintainer
**Related task:** TASK-15572
**Related plan:** [Server CI tiers](../Development/CI_REQUIRED_GATES.md#server-ci-tiers)

## Decision

Keep the existing required checks and license admission. When explicitly enabled, replace the supplemental Linux shard matrix with a bounded backend smoke job only for same-repository `dev` PRs that modify allowlisted chunking files. Unsupported changes, uncertain diffs and fork changes retain the Linux matrix. With the opt-in unset, preserve the existing `dev` routing.

Run the existing broad Linux matrix against the dispatched `dev` commit nightly. Skip the dispatch if that commit already has a successful or active nightly. A failed attempt may be retried. Existing specialized workflows keep their schedules and platform coverage.

A PR into `main` runs the complete existing server matrix and validates the built distributions on GitHub's PR merge commit. Rolling images, tagged images and PyPI uploads require that same qualification in their run at the event commit. Keep the top-level PyPI workflow and its environments for Trusted Publishing. Run the rolling image and eligible automatic PyPI upload from one qualified `main` push. Standalone MCP Unified has its own pipeline.

## Context

The server already has focused required checks and a larger supplemental matrix. The latter gives relevant backend PRs hundreds of Linux jobs while Windows and macOS mostly run on main pushes, release and manual CI. Package and image publication have separate triggers and can run before the full matrix reports.

## Alternatives considered

- Make every dev PR light. This would move feedback for security, migrations, shared code and unfamiliar paths out of review.
- Look up another workflow's successful status before publish. Reusable workflows keep the test and publish dependencies in the same run and avoid associating another commit or a skipped check with release qualification.

## Consequences

The allowlist is deliberately small. Its checks exercise mapped behavior and health contracts. Nightly preserves the broader Linux coverage. Release qualification may take longer than a current release upload because all required tests must actually finish first.

Before enabling the policy, install the default-branch schedule and verify a real nightly and a non-publishing full qualification. Bind the observed development and main PR check contexts to GitHub Actions in their respective protections without replacing existing checks. Clear `CI_TIERED_ENABLED` to restore existing dev routing. Publication gating is independent of that variable.
