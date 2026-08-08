---
id: TASK-12993
title: Implement Claims Jobs Stage 2A analytics exports
status: In Progress
created_date: 2026-08-08 21:36
labels:
- claims
- jobs
- implementation
priority: high
references:
- TASK-12989
- TASK-12990
- Docs/superpowers/specs/2026-08-08-claims-jobs-stage2a-analytics-exports-design.md
- Docs/superpowers/plans/2026-08-08-claims-jobs-stage2a-analytics-exports-implementation-plan.md
documentation:
- Docs/superpowers/specs/2026-08-08-claims-jobs-stage2a-analytics-exports-design.md
- Docs/superpowers/plans/2026-08-08-claims-jobs-stage2a-analytics-exports-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved Stage 2A implementation plan to move Claims analytics export execution onto the shared Jobs control plane behind an opt-in producer flag while preserving the synchronous fallback and keeping all queue lifecycle and administration in Jobs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Jobs-enabled analytics export requests return HTTP 202 with a durable Claims Job and queued owner-scoped artifact; Jobs-disabled requests retain synchronous HTTP 200 behavior.
- [ ] #2 Claims owns normalized export requests, deterministic bounded rendering, artifacts, reconciliation, retention, and downloads while Jobs exclusively owns execution lifecycle, retries, leases, cancellation, quarantine, status, and admin controls.
- [ ] #3 Jobs payloads are strict versioned ID-only contracts and Jobs results contain only non-sensitive export metadata.
- [ ] #4 SQLite and PostgreSQL schemas, migrations, owner-scoped operations, active/archive Jobs reads, and cross-owner denials are implemented and tested.
- [ ] #5 Worker retries use the persisted snapshot, can recover failed artifacts, cannot overwrite ready artifacts, and repair missing Job associations.
- [ ] #6 JSON and CSV output enforce row and byte bounds, stable ordering, CSV formula protection, safe filenames, and correct content types.
- [ ] #7 List and download behavior exposes separate artifact and read-only Job statuses, returns 409 for non-ready artifacts, and keeps missing/wrong-owner responses indistinguishable.
- [ ] #8 Reconciliation and retention are conservative when Jobs is unavailable and delete only eligible terminal artifacts after grace and retention.
- [ ] #9 Focused, regression, PostgreSQL, property, lint, compile, and Bandit verification gates pass with only fixture-reported environment skips.
- [ ] #10 No review-metrics aggregation, cluster rebuild, scheduler, Claims queue-control API, or request-level idempotency work enters Stage 2A.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute the 12 tasks in Docs/superpowers/plans/2026-08-08-claims-jobs-stage2a-analytics-exports-implementation-plan.md using test-driven development and subagent-driven development. Each implementation task receives specification-compliance review followed by code-quality review before the next task begins.
<!-- SECTION:PLAN:END -->

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
