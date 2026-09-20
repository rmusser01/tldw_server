---
id: TASK-13132
title: Add ACP dispatch recovery and monotonic execution evidence
status: To Do
created_date: 2026-08-24 17:39
dependencies:
- TASK-13129
labels:
- scheduled-tasks
- phase-4d
- acp
- jobs
- recovery
- cancellation
priority: High
references:
- Docs/superpowers/specs/2026-08-24-scheduled-tasks-phase4d-agent-task-execution-design.md
- Docs/superpowers/plans/2026-08-24-scheduled-tasks-phase4d-feasibility-gate-implementation-plan.md
updated_date: 2026-08-27 03:14
documentation:
- Docs/ADR/041-scheduled-agent-execution-feasibility.md
- Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.json
- Docs/Evidence/Scheduled_Agent_Execution/2026-08-24-phase4d0f-baseline.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add ACP adapter session creation idempotent by stable dispatch token, exact lookup after process loss, and a durable per-attempt monotonic evidence journal for terminal, timeout, pre-action approval, effect, and cancellation events. Preserve stable adapter tokens separately from Scheduled Tasks execution fences and prevent stale owners from mutating canonical state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Adapter session creation accepts an exact dispatch token idempotently and digest mismatch fails closed.
- [ ] #2 Process-loss recovery finds the exact adapter session/attempt by dispatch token without creating duplicate execution.
- [ ] #3 Terminal, timeout, approval, effect, and cancellation evidence is append-only and ordered by an adapter-owned monotonic sequence.
- [ ] #4 Cancellation races use the monotonic boundary rather than cross-store wall-clock order and unresolved evidence remains uncertain.
- [ ] #5 Existing interactive ACP sessions and generic Sandbox idempotency remain compatible and cannot expand scheduled authority.
- [ ] #6 Journal schema installation/upgrade, process restart, health, reconciliation, and partial-store outage behavior fail closed and publish bounded operator evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Phase 4D.0F evidence handoff: this task owns `adapter_dispatch_recovery` and `monotonic_execution_evidence`. Baseline evidence `sha256:1df8024b73472ea0a02a323fbad0d2f864d8b5f604611cb01bf49478f60a5874` records both as missing repository characterization; generic Sandbox idempotency and current cancellation primitives are dependencies, not the required dispatch-token and ordered per-attempt contracts. `operational_fail_closed` is a cross-cutting exit criterion: restarts, lease loss, timeout, cancellation, upgrades, and persistence outages must preserve exact recovery and monotonic evidence or disable execution.
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
