---
id: TASK-13153
title: Add atomic revision-guarded hard delete for Reading items
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03 02:27'
updated_date: '2026-09-05 20:21'
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
ADR required: yes; ADR path: backlog/decisions/003-reading-atomic-hard-delete.md (existing approved ADR). Approved spec/plan: Docs/superpowers/{specs,plans}/2026-09-05-reading-output-file-reservations*. Draft PR https://github.com/rmusser01/tldw_server/pull/2903 targets dev. Prior checkpoints through 8647861ddf covered staging/evidence and atomic recorded DB mutation. Current Task 3b copy/publication checkpoint verified: descriptor-relative bounded source copy, no-clobber link and directory fsync before recorded commit, fresh fenced outcome reads after uncertain acknowledgement, committed-wins conditional-abort handling. All files and reservations remain until phase-specific cleanup. TDD: 13 missing-method failures; EOF stage-check failure fixed; independent review reproduced delayed-commit false conflict, two RED regressions fixed and follow-up review confirmed resolution. Final SQLite/non-PG: 223 passed in 21.21s. Broader PG: 150 passed in 384.59s before final review fix; all 24 new PG copy/publication cases rerun after fix passed in 76.32s. 375 distinct targeted cases, no required-backend skips. Service/tests Ruff/Black, adapter touched-range Black, compile/diff and scoped Bandit pass; nine adapter Ruff baseline findings unchanged. Evidence/commands and lesson are in the plan and backlog/docs/lessons-testing-evidence.md. Commit/push this checkpoint to existing draft PR. Next: phase-specific cleanup/recovery with witness ordering, source fingerprint/reference checks, fs_done/retry handling and remaining crash matrix, then producer/reader integration and rollout gates. No merge, activation, full-suite or full-task completion claim; AC unchecked and In Progress.
<!-- SECTION:PLAN:END -->
