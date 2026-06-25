---
id: TASK-10005
title: Harden Scheduler module review findings
status: Done
assignee: []
created_date: 2026-06-23 21:55
updated_date: 2026-06-25 02:06
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix Scheduler review findings around queue status persistence, SQLite timestamp comparisons, lease-scoped acknowledgements, ACP trigger submission, Scheduler startup loops, authorization rate limits, and worker-pool drift.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PostgreSQL enqueue and bulk enqueue persist submitted tasks as queued.
- [x] #2 SQLite scheduled/deadline and lease-expiry checks parse ISO timestamps correctly.
- [x] #3 Worker ack/nack paths are lease-scoped so stale workers cannot complete reclaimed tasks.
- [x] #4 ACP webhook submissions await the global scheduler and include required Scheduler metadata.
- [x] #5 Scheduler startup loops, authorizer rate limits, and improved worker-pool drift have focused regression coverage.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Scheduler review hardening. Added red/green regression tests for startup loop races, SQLite ISO timestamp comparisons, stale lease ownership, PostgreSQL queued-state persistence, Scheduler authorizer rate limits, ACP webhook scheduler submission metadata, and the improved worker-pool release path. Adjusted the Unix fallback test to use a temporary home path so it does not write outside the workspace.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed all Scheduler review findings: PostgreSQL now persists new tasks as queued, SQLite timestamp predicates parse ISO strings, ack/nack can require matching lease and worker ownership, workers pass ownership tokens, ACP webhook submissions await the global scheduler and include metadata, startup loops are created after the scheduler is marked started, authorizer rate limits are enforced, and the improved worker pool no longer calls a nonexistent release_task method. Clean worktree verification: Scheduler tests plus ACP webhook trigger tests passed with 115 passed and 1 skipped; Bandit touched production scan reported 0 errors and 0 findings; staged and unstaged git diff --check both passed.
<!-- SECTION:FINAL_SUMMARY:END -->
