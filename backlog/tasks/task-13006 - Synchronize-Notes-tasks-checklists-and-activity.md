---
id: TASK-13006
title: Synchronize Notes tasks checklists and activity
status: Done
assignee: []
created_date: '2026-08-08 20:25'
updated_date: '2026-08-24 21:34'
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
- [x] #1 Capabilities advertise versioned `notes.task` and `notes.task_activity` domains using only their supported `upsert` and `tombstone` operations; immutable activity is created under a new stable event ID rather than a third wire operation.
- [x] #2 Task payloads preserve stable identity note linkage title description status priority due dates completion recurrence assignee tags metadata and optimistic base state across SQLite and PostgreSQL.
- [x] #3 Task activity preserves ordered immutable event identity actor timestamp transition and source without allowing history rewrites through ordinary updates.
- [x] #4 Markdown checklist projections are rebuilt deterministically and reconciliation reports drift without silently overwriting an explicit user-authored task.
- [x] #5 Server-origin task checklist reconciliation and activity mutations capture canonical envelopes when Sync v2 is active.
- [x] #6 Concurrent transitions recurrence completion delete restore and checklist edits yield idempotent results or reviewable conflicts with authorized bounded queries.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed TASK-13006.1 through TASK-13006.4 under ADR-039: tenant-scoped task storage and readiness, dormant notes.task and immutable notes.task_activity lifecycles, deterministic compound capture, managed checklist convergence, drift/retention repair, and coupled public activation. The final child added required multi-device SQLite/live-PostgreSQL convergence evidence and resolved every independent review finding.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The Notes task Sync workstream is complete. Final prescribed verification passed 433 tests with live PostgreSQL required and no skips; Ruff, Bandit, py_compile, and diff checks passed. Independent final review found no remaining actionable issues. Documentation and ADR references are current; known skips or blockers: none.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused Notes task activity recurrence and checklist projection suites pass on supported database backends.
- [x] #8 Bandit and static checks pass for touched production files.
- [x] #9 Event immutability reconciliation conflict and pagination scenarios have automated evidence.
<!-- DOD:END -->
