---
id: TASK-13217
title: Assess historical Jobs completion bookkeeping drift
status: To Do
created_date: 2026-09-07 20:06
dependencies:
- TASK-13215
labels:
- jobs
- data-integrity
- operations
priority: Medium
references:
- TASK-13215
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Assess whether deployments affected before TASK-13215 can retain lifecycle counter drift or missing job.completed events after the missing-row/concurrent-insert race. Define a bounded, operator-safe reconciliation path for counters and reporting guidance for event gaps without synthesizing outbox history when event enablement or provenance cannot be proven.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The assessment distinguishes counter drift from valid configurations where lifecycle counters or the events outbox were disabled.
- [ ] #2 A counter reconciliation procedure rebuilds scoped values from durable current job state and is safe to rerun.
- [ ] #3 Potential historical job.completed gaps are reported without inventing events when original outbox enablement or transition provenance is unknown.
- [ ] #4 Operator guidance states detection limits, locking/runtime impact, rollback, and verification.
- [ ] #5 Any implementation is covered on SQLite and required real PostgreSQL.
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
