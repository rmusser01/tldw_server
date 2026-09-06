---
id: TASK-13153
title: Add atomic revision-guarded hard delete for Reading items
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03 02:27'
updated_date: '2026-09-06 00:13'
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
ADR required: yes; ADR path: backlog/decisions/003-reading-atomic-hard-delete.md (existing). Continue approved Docs/superpowers/plans/2026-09-05-reading-output-file-reservations.md inline on draft PR 2903. Task 4c registered Watchlist readers implemented and reviewed: current metadata and descriptor transfer together, shared protected lookup, origin/expiry validation, current recipient plan for retry delivery, authenticated Collections adapter propagation, preserved text/audio HTTP behavior and descriptor cleanup. Activated text materialization is bounded at 8 MiB; inactive behavior unchanged. Metadata-only report sidecars deliberately fail closed for activated reads; Task 7 must establish reviewed producer/reconciliation provenance before Task 9 activation. Audit and exact tests/logs in plan checkpoint. Verification: 137 combined local cases, 10 existing output routes, three existing retry cases, five full-app report/evidence cases; broader PG run 99 passed plus one known failure from pre-final-patch worker code, replaced by a fresh passing targeted PG rerun (100 distinct PG cases; 255 overall, no backend skips). No claim that the broader PG invocation was clean. Service/new-test Ruff and Black, scoped Watchlist Black, compile/diff and Bandit pass; six pre-existing Watchlist Ruff findings unchanged. Review issues reproduced RED and fixed; bounded follow-up found no remaining actionable issues. Next implementation: Task 5 original-instance history delivery; sidecar provenance remains an explicit rollout blocker. No activation/background registration/full sweep/Docker provisioning/merge; task In Progress, AC unchecked, PR draft and human-written Change summary gate pending.
<!-- SECTION:PLAN:END -->
