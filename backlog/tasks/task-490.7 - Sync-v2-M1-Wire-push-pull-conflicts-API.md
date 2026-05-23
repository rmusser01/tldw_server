---
id: TASK-490.7
title: 'Sync v2 M1: Wire push pull conflicts API'
status: To Do
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m1
- api
- backend
priority: high
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wire the M1 materializer registry into Sync v2 push, pull, conflict, and legacy endpoint replacement behavior, including deterministic cursors, domain filters, pagination, echo handling, durable conflict resolutions, cross-user isolation, and replayable apply failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Push accepts, validates, persists, materializes, and reports apply outcomes per envelope.
- [ ] #2 Pull supports deterministic order, domain filters, pagination, has_more, next cursor, default echo suppression, and opt-in same-device echoes.
- [ ] #3 Conflict resolution records M1 actions without mutating historical envelopes.
- [ ] #4 Legacy /sync/send and /sync/get behavior is removed or clearly replaced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-7-wire-push-pull-conflicts-and-legacy-endpoint-replacement
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
