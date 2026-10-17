# Workflow WebSocket load benchmark — TASK-13820

Baseline: origin/dev0566130cbed2670efca6bbfa3c309031d2d3bc3b.
ADR check: no new ADR for a benchmark or local responsiveness repair; preserve auth, owner checks, frames and caching policy. ADR-059 governs tracking.

## Stage 1: Baseline and measurements
**Goal**: Measure production ASGI WebSocket path over real loopback TCP with isolated data.
**Success Criteria**: Reproducible setup/snapshot/event percentiles and correctness at1/16/64 clients, runtime/source identity, lifecycle cleanup and clear limits.
**Tests**: Harness validation/smoke behavior; baseline private app run.
**Status**: Complete

## Stage 2: Evidence-driven repair
**Goal**: Repair only reproduced root causes and compare matched workloads.
**Success Criteria**: Red/green regressions preserve auth, ownership, event delivery and safe diagnostics; paired before/after results.
**Tests**: Connected workflow/stream/auth suites and benchmark correctness/error cases.
**Status**: Complete

## Stage 3: Review and integration
**Goal**: Deliver a reviewed new PR against latest dev.
**Success Criteria**: Scoped lint/Bandit, independent exact-head review, requester Change summary and hosted checks before merge.
**Tests**: All hosted checks; latest-dev ancestry and merged source verification.
**Status**: In Progress
