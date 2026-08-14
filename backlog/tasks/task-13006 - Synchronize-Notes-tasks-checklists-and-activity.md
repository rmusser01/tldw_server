---
id: TASK-13006
title: Synchronize Notes tasks checklists and activity
status: In Progress
assignee: []
created_date: '2026-08-08 20:25'
updated_date: '2026-08-14 02:22'
labels:
  - notes
  - sync-v2
  - parity
  - tasks
dependencies:
  - TASK-13005
references:
  - >-
    https://github.com/rmusser01/tldw_chatbook/blob/dev/backlog/decisions/046-synchronized-database-notes-parity.md
  - Docs/ADR/039-canonical-notes-task-sync-and-derived-checklist-projections.md
documentation:
  - >-
    https://github.com/rmusser01/tldw_chatbook/blob/dev/Docs/Parity/2026-08-08-notes-server-capability-matrix.md
  - Docs/superpowers/specs/2026-08-13-notes-task-activity-sync-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give explicit Notes tasks, checklist state, and task activity a synchronized lifecycle so task workflows remain coherent offline and across clients while Markdown-derived task projections remain rebuildable rather than duplicated mutable state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Capabilities advertise versioned `notes.task` and `notes.task_activity` domains using only their supported `upsert` and `tombstone` operations; immutable activity is created under a new stable event ID rather than a third wire operation.
- [ ] #2 Task payloads preserve stable identity note linkage title description status priority due dates completion recurrence assignee tags metadata and optimistic base state across SQLite and PostgreSQL.
- [ ] #3 Task activity preserves ordered immutable event identity actor timestamp transition and source without allowing history rewrites through ordinary updates.
- [ ] #4 Markdown checklist projections are rebuilt deterministically and reconciliation reports drift without silently overwriting an explicit user-authored task.
- [ ] #5 Server-origin task checklist reconciliation and activity mutations capture canonical envelopes when Sync v2 is active.
- [ ] #6 Concurrent transitions recurrence completion delete restore and checklist edits yield idempotent results or reviewable conflicts with authorized bounded queries.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: Docs/ADR/039-canonical-notes-task-sync-and-derived-checklist-projections.md
Reason: ADR-039 already governs canonical task/activity authority, tenancy, compound mutation, checklist projection, and coupled activation.

1. Deliver TASK-13006.1 contract, schema v60, tenancy, and dormant readiness using Docs/superpowers/plans/2026-08-13-notes-task-sync-contract-storage-implementation-plan.md.
2. Deliver TASK-13006.2 dormant notes.task lifecycle using Docs/superpowers/plans/2026-08-13-notes-task-sync-lifecycle-implementation-plan.md.
3. Deliver TASK-13006.3 dormant notes.task_activity lifecycle using Docs/superpowers/plans/2026-08-13-notes-task-activity-sync-lifecycle-implementation-plan.md.
4. Deliver TASK-13006.4 immutable-envelope projection authority, client/server compound mutation, checklist convergence, coupled activation, and end-to-end proof using Docs/superpowers/plans/2026-08-13-notes-task-checklist-convergence-activation-implementation-plan.md.
5. Keep every child independently testable, require live PostgreSQL evidence, and expose neither task domain until the final child.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
- [ ] #7 Focused Notes task activity recurrence and checklist projection suites pass on supported database backends.
- [ ] #8 Bandit and static checks pass for touched production files.
- [ ] #9 Event immutability reconciliation conflict and pagination scenarios have automated evidence.
<!-- DOD:END -->
