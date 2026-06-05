---
id: TASK-2260
title: Backfill Services lifecycle startup and shutdown ADR
status: To Do
dependencies:
- TASK-2259
labels:
- docs
- process
- adr
- services
- lifecycle
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill a bounded Services lifecycle ADR from TASK-2259 evidence. Scope the accepted decision to FastAPI lifespan startup/shutdown orchestration through Services helpers, LifespanWorkerRuntimeState ownership of the worker lifecycle session, declarative lifecycle worker specs/engine/session ownership, cooperative stop-event workers with bounded timeout/cancel fallback, job-poller drain/quiesce before background worker shutdown, and explicit caveats for callback-only workers, legacy shutdown adapters, and scope limited to lifecycle-managed Services workers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Create the next accepted ADR under `Docs/ADR/` using the standard ADR template and TASK-2259 evidence.
- [ ] #2 Keep accepted claims scoped to Services lifespan orchestration, worker lifecycle session ownership, declarative worker specs/engine/session, stop-event default strategy, staged shutdown order, and documented caveats.
- [ ] #3 Update `Docs/ADR/README.md`, the INV-031 inventory row, and the Services README backlink after ADR creation.
- [ ] #4 Record verification and Bandit applicability in this task.
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
