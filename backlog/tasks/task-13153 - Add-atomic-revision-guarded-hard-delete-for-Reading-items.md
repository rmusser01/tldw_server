---
id: TASK-13153
title: Add atomic revision-guarded hard delete for Reading items
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03 02:27'
updated_date: '2026-09-06 01:22'
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
ADR required: yes; ADR path: backlog/decisions/003-reading-atomic-hard-delete.md. Approved output-file-reservations plan continues inline. Task 5a receiver foundation verified: Media v27 original-instance live/disposed receiver, both migrations, late-insert and reused-ID/user isolation, first disposal evidence, caller transaction ownership including mixed legacy inserts, optional/partial schema recovery and SQLite pooled-handle migration fix. Verification: 34 real SQLite/PostgreSQL receiver cases, 97 existing local schema/history/purge cases, 5 required PostgreSQL schema cases and 62 TTS consumer cases passed (198 distinct); no required backend skips. Scoped static/security checks pass with one unchanged SQLite bootstrap I001 baseline. Independent review transaction finding reproduced RED and fixed; follow-up cleared it. Next Task 5b: RED post-fs_done outage/lost-ack/recycled-ID tests, original-incarnation capture at TTS producer time, bounded durable delivery outside filesystem exclusion, independent backoff and conditional acknowledgement. No runtime activation, full sweep, Docker or merge. Full task In Progress, ACs unchecked, PR 2903 draft; human Change summary gate pending.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 5a checkpoint: implemented and independently reviewed original-instance Media history disposal receiver, schema v27 migrations and transaction ownership fixes. Existing ADR-003 applies; detailed evidence and remaining Task 5b are in Docs/superpowers/plans/2026-09-05-reading-output-file-reservations.md. Targeted evidence: 198 distinct passing cases across receiver, migration, history, TTS worker/endpoint/cleanup and pagination suites; 22 real PostgreSQL cases included, none skipped in their final required-backend runs. Production Bandit zero findings/errors; test Bandit excludes only assertions; changed-range Black/compile/diff checks pass. No new Ruff findings; one unchanged SQLite bootstrap import-order baseline remains. No activation or full-task completion claimed.
<!-- SECTION:NOTES:END -->
