---
id: TASK-490.8
title: 'Sync v2 M1: Route server-origin Notes and Chat through Sync'
status: To Do
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m1
- server-frontend
- backend
priority: high
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Route personal Notes and Chat mutations made through normal server APIs through Sync v2 when Sync is active so server-front-end writes are represented in the append-only envelope log before materialized projections exist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Personal Notes and Chat server API writes create server-origin envelopes before projection writes occur.
- [ ] #2 Envelope append failures prevent or roll back the normal API mutation so projections cannot exist without log entries.
- [ ] #3 Materialization failures leave replayable failed envelopes and are visible in profile status.
- [ ] #4 Offline-sync devices can pull server-origin envelopes by cursor/domain filter.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-8-route-server-origin-notes-and-chat-changes-through-sync
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
