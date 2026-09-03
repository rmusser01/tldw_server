---
id: TASK-13153
title: Add atomic revision-guarded hard delete for Reading items
status: To Do
assignee: []
created_date: '2026-09-03 02:27'
updated_date: '2026-09-03 02:29'
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
Add a positive, monotonically increasing integer `revision` to persisted Reading items and every Reading summary/detail response. Every item-owned mutation that can change the user's deletion decision increments that revision, including metadata/status/tag changes and capture-owned content changes. The permanent-delete request accepts `expected_revision` and enforces the authenticated user, item, and revision predicate in the same transaction as child cleanup and deletion. Stale preconditions conflict without deletion; missing items remain distinct. Delete capture-owned children/artifacts but never linked external Media or Notes. Advertise exact `hasReadingOptimisticDeletesV1=true` only when the complete atomic contract is active.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Reading item summary/detail exposes a positive monotonic `revision`, and every item-owned change relevant to deletion advances it in SQLite and PostgreSQL.
- [ ] #2 Hard delete atomically requires the authenticated user's exact `expected_revision`; stale requests return a documented conflict and remove nothing.
- [ ] #3 Confirmed hard delete removes the Reading item and capture-owned children/artifacts without deleting linked external Media or Notes.
- [ ] #4 Schema migration, concurrent mutation/delete, wrong-user, missing-item, rollback, and cascade behavior have focused SQLite/PostgreSQL regression coverage.
- [ ] #5 Docs-info advertises `hasReadingOptimisticDeletesV1=true` only when the complete contract is active, and public API documentation describes the precondition and responses.
- [ ] #6 A new or applicable Server ADR records the schema, destructive precondition, and ownership decision before implementation begins.
<!-- AC:END -->
