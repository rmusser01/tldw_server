---
id: TASK-13000
title: Fix audit original file storage cleanup on DB registration failure
status: Done
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
updated_date: 2026-07-18 18:03
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
- [x] #5 All PR review threads and current-dev integration effects are reconciled.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: Migrate colliding task IDs and rebase PR #2612 onto current origin/dev. Stage 2: verify both existing review fixes and inspect the full latest-dev diff for remaining cleanup hazards. Stage 3: add failing regression tests and minimal fixes for any confirmed gaps. Stage 4: run focused verification, independent reviews, push, resolve threads, and record CI state.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reserved TASK-13000 after latest dev consumed the transient TASK-12947 ID. The transient media TASK-12947 record was removed because latest dev already owns that active ID; an accidental duplicate-ID edit to the unrelated upstream browser task was immediately restored to a zero diff. The original media TASK-12145 record is archived with its history.

Verified on origin/dev f05fe296: all 16 focused original-storage tests pass, including generic registration failure combined with generic cleanup failure. Targeted Ruff BLE001 checks pass; whole-file Ruff reports 51 pre-existing unrelated diagnostics outside this PR's scope. Bandit reports 0 findings over the touched production file, and diff/whitespace checks are clean. Independent specification review approved after strengthening the critical-failure regression test.

Independent final code-quality/security review approved. Residual risk: task cancellation during storage.delete can supersede the original registration exception; this is accepted because cancellation must retain propagation semantics, while ordinary cleanup failures are caught and tested.

Final GitHub reconciliation: both requested review threads are resolved and no other actionable comment remains. Fresh final-head workflows were triggered; decisive jobs were queued with no reported failure at the reconciliation checkpoint, matching the repository-wide runner backlog already documented during this audit.
Latest-dev refresh on 2026-07-18: rebased again after dev advanced to 668b0fce5707134768f880b5d064ccc5b0cc4691. Post-rebase verification remains clean: 16 focused tests pass, targeted Ruff passes, Bandit reports 0 findings and 0 errors over 5,739 LOC, diff checks pass, and merge-base equals the fetched dev tip.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2612 onto dev 668b0fce, verified broad registration and cleanup exception handling, and strengthened regression coverage so a generic cleanup failure cannot mask a generic registration failure. All 16 focused tests pass; targeted Ruff and diff checks pass; Bandit reports 0 findings; independent specification and quality/security reviews approved. Migrated the colliding audit task history to TASK-13000, resolved both GitHub review threads, refreshed the PR description, and documented the fresh queued CI state.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused original-storage tests pass on latest dev.
- [x] #2 Applicable Ruff checks pass or pre-existing unrelated diagnostics are documented.
- [x] #3 Bandit reports no findings in touched production scope.
- [x] #4 git diff --check passes.
- [x] #5 Independent specification and code-quality reviews have no unresolved actionable findings.
- [x] #6 PR review threads and fresh CI state are reconciled.
<!-- DOD:END -->
