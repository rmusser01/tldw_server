---
id: TASK-13517
title: Stop event-loop blocking in workers, schedulers, destructive endpoints, and
  MCP memory growth
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Batch 5. Plan: Docs/Plans/2026-10-06-perf-batch-5-event-loop-workers-implementation-plan.md. Worker acquire tick via to_thread + single connection + drop duplicate reconcile (Jobs/manager.py:6159); SLO gauges from histograms at lower cadence (jobs_metrics_service.py:271); workflow pause backoff/event (Workflows/engine.py:515); diff-based scheduler rescans + shared user enumeration; bulk delete_chat_session; chunked empty-trash; export and attachment sync I/O off loop; watchlist alert-rules connection reuse; MCP metrics label whitelist + summary from counters; JWT manager token pruning.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

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
