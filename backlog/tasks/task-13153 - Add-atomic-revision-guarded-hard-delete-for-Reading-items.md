---
id: TASK-13153
title: Add atomic revision-guarded hard delete for Reading items
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03 02:27'
updated_date: '2026-09-05 19:59'
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
ADR required: yes; ADR path: backlog/decisions/003-reading-atomic-hard-delete.md. Approved spec/plan: Docs/superpowers/{specs,plans}/2026-09-05-reading-output-file-reservations*. Draft PR https://github.com/rmusser01/tldw_server/pull/2903 against dev opened at f43549c209 before continuing, as requested. Task 3b DB-owned recorded-commit boundary verified: apply_output_file_operation revalidates authority/evidence, applies recorded create/replace/remove, guards allocated IDs, and commits publication identity, phase, byte accounting and original-instance removal history on one connection. Replay rejects before additive accounting; zero affected output rows roll back; missing/invalid audiobook usage fails closed. No filesystem proof or runtime caller is supplied by this DB-only method. Evidence: 199 final SQLite/non-PostgreSQL passes (24.28s), 126 broader PostgreSQL passes (342.31s), then all 25 focused PostgreSQL recorded-commit cases passed after final rowcount guard (76.54s). 327 distinct cases total; overlapping reruns not double-counted; no required-backend skips. Independent review and final guard follow-up found no serious concerns. Modified test Ruff/Black, touched adapter-range Black, compile/diff and scoped Bandit pass; nine preexisting adapter Ruff findings remain. Commit/push this checkpoint to draft PR. Next: descriptor-relative copy, no-clobber publication, uncertain commit acknowledgement and phase-specific restart recovery; producer metadata/idempotency integration and rollout gates remain pending. No merge, activation, full-suite or full-task completion claim; AC unchecked and In Progress.
<!-- SECTION:PLAN:END -->
