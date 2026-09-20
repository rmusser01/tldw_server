---
id: TASK-13000
title: Fix audit original file storage cleanup on DB registration failure
status: Done
assignee: []
created_date: '2026-07-14 07:05'
updated_date: '2026-09-10 00:50'
labels:
  - audit
  - remediation
  - media
  - storage
  - pr-followup
dependencies: []
references:
  - AUDIT-2026-06-27-MEDIA-003
  - 'https://github.com/rmusser01/tldw_server/pull/2612'
  - Supersedes colliding media task records TASK-12145 and TASK-12947
documentation:
  - >-
    Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md
  - Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md
priority: medium
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

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reserved TASK-13000 after latest dev consumed the transient TASK-12947 ID. The transient media TASK-12947 record was removed because latest dev already owns that active ID; an accidental duplicate-ID edit to the unrelated upstream browser task was immediately restored to a zero diff. The original media TASK-12145 record is archived with its history.

Verified on origin/dev f05fe296: all 16 focused original-storage tests pass, including generic registration failure combined with generic cleanup failure. Targeted Ruff BLE001 checks pass; whole-file Ruff reports 51 pre-existing unrelated diagnostics outside this PR's scope. Bandit reports 0 findings over the touched production file, and diff/whitespace checks are clean. Independent specification review approved after strengthening the critical-failure regression test.

Independent final code-quality/security review approved. Residual risk: task cancellation during storage.delete can supersede the original registration exception; this is accepted because cancellation must retain propagation semantics, while ordinary cleanup failures are caught and tested.

Final GitHub reconciliation: both requested review threads are resolved and no other actionable comment remains. Fresh final-head workflows were triggered; decisive jobs were queued with no reported failure at the reconciliation checkpoint, matching the repository-wide runner backlog already documented during this audit.
Latest-dev refresh on 2026-07-18: rebased again after dev advanced to 668b0fce5707134768f880b5d064ccc5b0cc4691. Post-rebase verification remains clean: 16 focused tests pass, targeted Ruff passes, Bandit reports 0 findings and 0 errors over 5,739 LOC, diff checks pass, and merge-base equals the fetched dev tip.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-09-09: Reopened for user-requested PR #2612 review and rebase onto origin/dev 40345571a2cfc8b3a8893545836097d27e4ee86c. Plan: verify continued applicability and historical review threads; rebase in isolated worktree; reproduce the failure on dev and verify the fixed behavior; run focused and adjacent tests, Ruff, Bandit, and independent review; update the PR branch and record remaining merge gates.

2026-09-09 review findings: all four PR regression cases fail against current dev and pass after rebasing. Independent review found P1 shared-path deletion on repeated ingestion; storage must allocate a distinct blob per registration attempt, and retrieval must select the latest registered file to preserve replacement behavior. Cancellation during deletion is swallowed by three enclosing broad handlers; add explicit propagation with regression coverage. Combined tests also exposed a cached upload-quota service leaking into this unit module; isolate its existing fake quota service. Full Ruff comparison found five new legacy typing diagnostics in PR-added tests; fix them. Existing unrelated diagnostics remain out of scope.

Review remediation: generate a UUID storage filename for each original registration and select the latest matching MediaFiles row by descending id. Real SQLite/filesystem regressions reproduced legacy-path deletion, overwrite damage, and stale retrieval, then passed with the fix. Explicit cancellation propagation through the three enclosing orchestrator handlers fixes swallowed cleanup cancellation. A per-test upload-quota fake fixes suite-order leakage. First adjacent matrix passed 131 tests, including original storage, all persistence unit modules, filesystem/interface, MediaFiles DB operations, file endpoints, and cleanup service. Final expanded matrix and independent re-review are in progress. Retention tradeoff: earlier successful original blobs remain registered and are retained; bounded retention and quota accounting are explicitly tracked in TASK-13000.1. No new retention or quota-management guarantee is claimed.

2026-09-09: Superseded by TASK-13233 because current dev owns TASK-13000 for UserProfiles work. This media cleanup history is archived; active review and verification are tracked under TASK-13233, and retention/quota follow-up under TASK-13233.1.
<!-- SECTION:NOTES:END -->

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
