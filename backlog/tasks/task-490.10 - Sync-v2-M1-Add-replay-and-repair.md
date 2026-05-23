---
id: TASK-490.10
title: 'Sync v2 M1: Add replay and repair'
status: To Do
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m1
- repair
- backend
priority: high
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add replay/repair support that rebuilds materialized Notes and Chat projections from accepted envelopes, retries failed applies, preserves tombstones, excludes conflict envelopes, and reports repair status.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Replay can rebuild Notes and Chat projections from accepted envelopes.
- [ ] #2 Failed applies can be retried after the underlying projection issue is fixed.
- [ ] #3 Tombstones are preserved and conflict envelopes are not replayed as accepted changes.
- [ ] #4 Profile/status exposes failed apply counts and repair results.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-10-add-replay-and-repair
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
