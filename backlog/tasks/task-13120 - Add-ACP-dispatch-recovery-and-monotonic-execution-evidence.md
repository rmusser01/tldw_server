---
id: TASK-13120
title: Add ACP dispatch recovery and monotonic execution evidence
status: To Do
created_date: 2026-08-24 17:39
dependencies:
- TASK-13117
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
updated_date: 2026-08-24 17:55
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
