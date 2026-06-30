---
id: TASK-490.4
title: 'Sync v2 M1: Materialize Notes'
status: Done
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m1
- notes
- backend
priority: high
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Notes domain materialization for notes.note upserts and tombstones through DB_Management-owned ChaChaNotes helpers, including whole-object conflict detection, object state updates, apply status, and replayable failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 notes.note upserts create/update normal server notes visible through ChaChaNotes APIs.
- [x] #2 Stale base revisions or hashes create whole-object conflicts without overwriting projections.
- [x] #3 Tombstones soft-delete notes and stale upserts cannot resurrect deleted notes.
- [x] #4 Notes materializer and ChaChaNotes helper tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-4-implement-notes-materialization
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented DB_Management-owned ChaChaNotes sync helpers `upsert_note_from_sync` and `tombstone_note_from_sync`; added Sync v2 notes materializer for `notes.note` upsert/tombstone projection, whole-object base-state conflict detection, SyncObjectState updates, apply status updates, and failed-apply replayability; wired the per-user factory to provide a notes materializer; added focused Sync materializer and ChaChaNotes helper coverage. Amended after controller verification to make exact retries of already-applied note create/update/tombstone envelopes idempotent without re-materializing or mutating apply status, while preserving retry materialization for failed accepted envelopes. Amended after quality review to keep materialization-conflict envelopes out of normal pull visibility and to make exact retries of partially materialized failed envelopes complete idempotently when object state already reflects the same server cursor/hash/revision/deleted state. Amended after spec/quality re-review to preserve pull progress across hidden materialization-conflict envelopes by basing `has_more` on raw scanned rows and advancing `next_cursor` past suppressed conflicts when no visible lookahead would be skipped.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Materialized `notes.note` envelopes into normal ChaChaNotes notes. Clean upserts create/update visible notes, stale base revision/hash inputs create whole-object conflicts without overwriting projections, tombstones soft-delete notes, stale upserts cannot resurrect tombstoned notes, projection failures mark accepted envelopes failed for replay, exact retries of already-applied create/update/tombstone envelopes remain applied without false conflicts, materialization conflicts are not returned by normal pull, exact retries after object-state update plus failed applied-status marking complete without self-conflict, and pull pagination advances through hidden materialization-conflict rows without looping or stranding later visible envelopes. Verification: pagination regression test failed before fix (`has_more=False`, `next_cursor='1'`), then passed after fix (1 passed); focused notes/materializer tests passed (28 passed); requested Sync v2 regression smoke passed (88 passed); domain adapter check passed (27 passed); Bandit on touched production scope returned 0 findings.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
