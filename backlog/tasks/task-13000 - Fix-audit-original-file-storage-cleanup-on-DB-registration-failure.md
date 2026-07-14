---
id: TASK-13000
title: Fix audit original file storage cleanup on DB registration failure
status: In Progress
created_date: 2026-07-14 07:05
labels:
- audit
- remediation
- media
- storage
- pr-followup
priority: medium
references:
- AUDIT-2026-06-27-MEDIA-003
- https://github.com/rmusser01/tldw_server/pull/2612
- Supersedes colliding media task records TASK-12145 and TASK-12947
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md
modified_files:
- tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py
updated_date: 2026-07-14 15:52
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete AUDIT-2026-06-27-MEDIA-003 and PR #2612 follow-up by preserving compensating deletion when original-file registration fails, verifying broad registration and cleanup exceptions do not orphan files or mask the original failure, and refreshing the branch onto current dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When permanent original storage succeeds but media-file registration raises an ordinary Exception, compensating deletion is attempted.
- [x] #2 Cleanup failures, including ordinary Exception and false return values, are logged without masking the original registration failure.
- [x] #3 The affected result reports original_file_stored false and exposes no retrievable original_file_path.
- [x] #4 Focused tests cover successful cleanup, cleanup exceptions, cleanup false returns, and successful storage registration.
- [ ] #5 All PR review threads and current-dev integration effects are reconciled.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: Migrate colliding task IDs and rebase PR #2612 onto current origin/dev. Stage 2: verify both existing review fixes and inspect the full latest-dev diff for remaining cleanup hazards. Stage 3: add failing regression tests and minimal fixes for any confirmed gaps. Stage 4: run focused verification, independent reviews, push, resolve threads, and record CI state.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reserved TASK-13000 after dev advanced during the first rebase and consumed TASK-12947. Earlier media records TASK-12145 and TASK-12947 are archived as superseded. Existing GitHub comments requested broad Exception handling for registration and cleanup failures; the branch claims both were implemented and must be re-verified on current dev.
Correction: the transient media TASK-12947 record was removed rather than archived because latest dev already owns that active ID and duplicate-ID Backlog lookup targeted the unrelated browser task. That accidental upstream-task mutation was immediately restored to a zero diff. The original media TASK-12145 remains archived with its history.
Verified on origin/dev f05fe296: all 16 focused original-storage tests pass, including generic registration failure combined with generic cleanup failure. Targeted Ruff BLE001 checks pass; whole-file Ruff reports 51 pre-existing unrelated diagnostics outside this PR's scope. Bandit reports 0 findings over the touched production file, and diff/whitespace checks are clean. Independent specification review approved after strengthening the critical-failure regression test.
Independent final code-quality/security review approved. Residual risk: task cancellation during storage.delete can supersede the original registration exception; this is accepted because cancellation must retain propagation semantics, while ordinary cleanup failures are caught and tested.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused original-storage tests pass on latest dev.
- [x] #2 Applicable Ruff checks pass or pre-existing unrelated diagnostics are documented.
- [x] #3 Bandit reports no findings in touched production scope.
- [x] #4 git diff --check passes.
- [x] #5 Independent specification and code-quality reviews have no unresolved actionable findings.
- [ ] #6 PR review threads and fresh CI state are reconciled.
<!-- DOD:END -->
