---
id: TASK-12100
title: Fix media ingest worker startup default and capability reporting
status: In Progress
labels:
- backend
- jobs
- media-ingest
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement fixes for media ingest jobs staying queued when the normal media ingest worker is not started by default, plus incorrect hasMediaIngestWorker capability reporting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design spec written: Docs/superpowers/specs/2026-07-09-media-ingest-worker-startup-capability-design.md
Implementation plan written: Docs/superpowers/plans/2026-07-09-media-ingest-worker-startup-capability-implementation-plan.md
Spec review: Approved locally. No TODO/TBD/placeholders; scope limited to media ingest worker startup, config capability reporting, and matching docs. Subagent reviewer not dispatched because this session's tool rules require explicit user authorization for delegation.
Plan review: Approved locally after correcting the route-policy design to inject WorkerLifecycleContext.route_enabled into should_start_inprocess_worker(). No implementation code changed yet.
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
