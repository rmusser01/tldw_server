# Workflow SQLite benchmark and WebSocket diagnostics — TASK-13819

Baseline: origin/dev 696d7fdeced92f23b72ae416aeddc6ea9fd5f945, including PR #3252.
ADR required: no; measure the existing cache-disable repair and improve local diagnostic observability without changing storage or authorization policy. ADR-059 governs task tracking.

## Stage 1: Cleanup and baseline
**Goal**: Retire task-owned chat worktrees safely and establish isolated latest-dev baseline.
**Success Criteria**: Preserve runtime evidence privately; archive retired worktrees; leave primary checkout and other sessions untouched.
**Tests**: Inspect attached identities, merged PRs, worktree state and owned processes before cleanup.
**Status**: Complete

## Stage 2: Measure and diagnose
**Goal**: Quantify cached-versus-uncached real workflow read costs and make WebSocket refusals diagnosable.
**Success Criteria**: Reproducible paired timing and correctness measurements; safe stable rejection reason/close code; no credentials or private run data in diagnostics.
**Tests**: Benchmark smoke/validation tests, real database identity checks, failing-first endpoint diagnostic tests and existing ownership/cookie cases.
**Status**: Complete

## Stage 3: Verify and integrate
**Goal**: Review and merge a new PR against latest dev after all checks pass.
**Success Criteria**: Connected tests, scoped Bandit/lint, independent exact-head review, requester Change summary and hosted gates pass before merge.
**Tests**: Hosted workflow storage/runtime shards, all PR checks, remote-dev merge ancestry.
**Status**: In Progress
