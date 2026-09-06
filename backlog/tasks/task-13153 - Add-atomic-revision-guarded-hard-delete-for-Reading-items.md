---
id: TASK-13153
title: Add atomic revision-guarded hard delete for Reading items
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03 02:27'
updated_date: '2026-09-06 03:55'
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
ADR required: yes; existing backlog/decisions/003-reading-atomic-hard-delete.md contains the DB evidence amendment. Direction, lifecycle and compatibility approved. Written spec Docs/superpowers/specs/2026-09-05-reading-report-evidence-database-design.md awaits final user review. After approval, use writing-plans for bounded storage/deletion, producer/read and legacy migration checkpoints, preserving the broader Task7 writer inventory. No production changes, activation, capability advertisement, merge or full sweep at this design checkpoint.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 5a checkpoint: implemented and independently reviewed original-instance Media history disposal receiver, schema v27 migrations and transaction ownership fixes. Existing ADR-003 applies; detailed evidence and remaining Task 5b are in Docs/superpowers/plans/2026-09-05-reading-output-file-reservations.md. Targeted evidence: 198 distinct passing cases across receiver, migration, history, TTS worker/endpoint/cleanup and pagination suites; 22 real PostgreSQL cases included, none skipped in their final required-backend runs. Production Bandit zero findings/errors; test Bandit excludes only assertions; changed-range Black/compile/diff checks pass. No new Ruff findings; one unchanged SQLite bootstrap import-order baseline remains. No activation or full-task completion claimed.

Task 5b checkpoint: added bounded, filesystem-independent delivery of original-instance history effects with separate backoff/operator blocks, cancellation draining, replay-safe acknowledgement and retirement. TTS jobs capture incarnation through one creation transaction without public DTO changes; synchronous speech remains storage-file-ID-only. Independent review exposed a PostgreSQL nested-connection gap; real RED rollback/NOWAIT fence regressions now pass with explicit connection reuse. Existing ADR-003 applies; plan and testing lesson updated. Fresh targeted verification: 340 distinct passes (191 SQLite/non-PostgreSQL, 18 PostgreSQL sender, 131 PostgreSQL journal/recovery/receiver), with only complementary backend-specific skips. Changed-scope formatting, compile/diff checks and Bandit pass; no new Ruff findings, baseline 9 adapter + 1 worker-test warnings unchanged. No Docker, full sweep, activation or merge. Next is Task 6 PATCH/deletion/retention integration; full task stays In Progress with ACs unchecked and human Change summary pending.

Task 6a checkpoint: protected compound PATCH now reserves/copies or bounds conversion input/output, publishes without clobber, and commits final title/format/retention together. Reuses the existing journal and exact-source byte admission; no intermediate rename or activated fallback. Preserves managed metadata/conversion policy, case-only/no-op metadata behavior, shared-unowned sources, inactive legacy dispatch, and original rows/bytes on rejection. Non-creating root resolution rejects missing volumes. Cancellation aborts producer authority before a delayed renderer can publish. Corrected a late-owner fixture that had rejected registration before the race; an in-memory bypass then reproduced the destructive legacy rename, and all six strengthened race cases passed on SQLite/PostgreSQL. One old consumer double now declares real user/inactive binding. Fresh targeted evidence: 187 distinct passing cases with no backend skips; scoped formatting/compile/diff and Bandit pass; no new Ruff findings (six existing baseline findings unchanged). Independent read-only review found no actionable issue. Existing ADR-003 applies, detailed evidence in the output-file-reservations plan. Task 6b deletion/purge integration remains next. No full sweep, Docker, activation, capability advertisement or merge; full task In Progress and human Change summary pending.

Task 6b checkpoint: activated unowned DELETE and both retention purge paths now use journal-owned removal with atomic quota/original-incarnation history effects. Managed Reading dispatch reuses its transaction connection; metadata-only and genuine inactive behavior remain. Eligibility rechecks under preparation fence prevent stale-scan renewal deletion. Actual synced-unlink receipts survive later acknowledgement failure, while shared/deferred files are not counted. Added busy409 and offline missing404 regressions. Existing ADR-003 applies; independent read-only review and bounded rereviews have no outstanding findings. Targeted evidence: 290 distinct passes across SQLite/PostgreSQL; final24 boundary rerun overlaps. Production Bandit clean; test-security5 and Ruff16 findings match HEAD baseline; scoped format/compile/diff pass. Details in the output-file-reservations plan. Task7 producer integration is next. No full sweep, Docker, activation, capability advertisement or merge; full task stays In Progress with ACs unchecked and human Change summary pending.

Task7 preflight from6b1465607f: traced generic/Watchlist generation, file-first failure cleanup, Reading archive creation, evidence sidecars and output-to-Media ingestion. Confirmed unregistered report_snapshot_path JSON has no shared ownership/deletion authority; choosing DB-backed evidence versus owned files requires architectural approval and explicit legacy compatibility. Additional top-level writers found in Research_Workspace/output_jobs.py and presentation_render_jobs_worker.py; other root users require boundary classification before activation. Inventory and exact next decision recorded in the output-file-reservations plan. No producer/runtime changes or tests run; implementation paused at the brainstorming architectural approval gate, not marked blocked/Done.

User approved the DB-backed immutable evidence direction. Architectural brainstorming now covers one bounded immutable snapshot shared through explicit output-variant links, transactional deletion and legacy reconciliation. Detailed lifecycle/spec not yet approved; no production code changes. Inspection also confirms variant metadata is copied before the primary report snapshot metadata is added, so current variant linkage must not be assumed from filenames or variant_of alone.

User approved evidence lifecycle: one immutable snapshot per generation with explicit variant links; soft deletion retains links, hard deletion removes its link and atomically deletes the last-unreferenced snapshot; evidence reads need no output volume; legacy migration is dry-run-first with unchanged-record checks and preserves source files. Next design section defines finite resource limits, API compatibility and failure behavior before writing the spec/ADR amendment. No implementation yet.

User approved the compatibility/resource/failure section. Wrote Docs/superpowers/specs/2026-09-05-reading-report-evidence-database-design.md and linked the proposed written amendment from ADR003 and the parent plan. Self-review clarified output-incarnation cleanup guards, delayed-variant rejection, preallocated evidence digest versus historical primary output ID, DB-only import replay, structural missing-link detection, and truly non-initializing dry runs. Written spec awaits user review; implementation planning/code has not started. Documentation-only checks: diff whitespace and placeholder/link inspection; no tests or activation.
<!-- SECTION:NOTES:END -->
