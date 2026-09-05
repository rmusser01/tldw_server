---
id: TASK-13153
title: Add atomic revision-guarded hard delete for Reading items
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03 02:27'
updated_date: '2026-09-05 03:15'
labels:
  - collections
  - reading-list
  - api
  - concurrency
  - deletion
dependencies: []
references:
  - 'tldw_chatbook:TASK-18919'
  - >-
    tldw_chatbook:Docs/superpowers/specs/2026-09-01-collections-followup-backlog-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a positive, monotonically increasing integer `revision` to persisted Reading items and every
Reading summary/detail response. Every item-owned mutation that can change the user's deletion
decision increments that revision, including metadata/status/tag changes and capture-owned content
changes. The permanent-delete request accepts `expected_revision` and enforces `WHERE id = ? AND
user_id = ? AND revision = ?` or its backend-equivalent in the same transaction as child cleanup
and deletion. A stale precondition returns a documented 409/412 conflict without deleting the item;
missing items remain distinct. The operation removes capture-owned children and artifacts but does
not delete linked external Media or Notes. Docs-info advertises exact
`hasReadingOptimisticDeletesV1=true` only when the response token and atomic mutation are active.
SQLite/PostgreSQL migration, concurrent mutation/delete, authorization, conflict, cascade, and
diagnostic-privacy behavior are covered.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Reading item summary/detail exposes a positive monotonic `revision`, and every item-owned change relevant to deletion advances it in SQLite and PostgreSQL.
- [ ] #2 Hard delete atomically requires the authenticated user's exact `expected_revision`; stale requests return a documented conflict and remove nothing.
- [ ] #3 Confirmed hard delete removes the Reading item and capture-owned children/artifacts without deleting linked external Media or Notes.
- [ ] #4 Schema migration, concurrent mutation/delete, wrong-user, missing-item, rollback, and cascade behavior have focused SQLite/PostgreSQL regression coverage.
- [ ] #5 Docs-info advertises `hasReadingOptimisticDeletesV1=true` only when the complete contract is active, and public API documentation describes the precondition and responses.
- [ ] #6 A new or applicable Server ADR records the schema, destructive precondition, and ownership decision before implementation begins.
- [ ] #7 Late or crashed artifact writers cannot create untracked files after cleanup; cleanup verifies the owning storage namespace and filesystem exclusion before treating absence as success.
- [ ] #8 Legitimate legacy manual and older archives have a documented dry-run-first reconciliation path with unchanged-record checks and no deletion.
- [ ] #9 Optimistic-delete capability defaults false when readiness is unknown or unavailable, and direct hard-delete requests independently reject unavailable target stores without mutation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/003-reading-atomic-hard-delete.md. Reason: persisted aggregate revisions, destructive preconditions, ownership and durable cleanup. Design: Docs/superpowers/specs/2026-09-04-reading-atomic-hard-delete-design.md. Plan: Docs/superpowers/plans/2026-09-04-reading-atomic-hard-delete.md. Stage 1 clock/item/tag/link/highlight/output metadata checkpoints remain partial. Stage 3 now includes durable unadopted staging/pending paths, original-revision validation before exclusive write, bounded retry cleanup under verified POSIX storage lock, descriptor-relative I/O and directory-sync-before-retirement. Generic output create/path changes fence reserved paths; references and protected names handle ASCII case aliases. Review reproduced and fixed replacement-volume and alias hazards; re-review clear. Verification: 43 SQLite/storage cases; all 20 real PostgreSQL lifecycle cases covered in 17+5 passes with two overlaps, no skips; 123 existing regression cases. Scoped Bandit zero; new service/tests lint/format clean; DB keeps nine baseline findings. Details and lessons are in plan/docs. Next: guarded adoption, owned-output/hard-delete intent creation, legacy reconciliation, purge routing, DTO snapshots and startup readiness. No production service caller, ownership activation or capability yet. Task remains In Progress.
<!-- SECTION:PLAN:END -->
