---
id: TASK-13001
title: Refresh audit oversized audio download regression PR
status: In Progress
created_date: 2026-07-18 18:35
labels:
- audit
- remediation
- media
- tests
- pr-followup
priority: low
references:
- AUDIT-2026-06-27-MEDIA-004
- https://github.com/rmusser01/tldw_server/pull/2613
- Supersedes colliding audit task TASK-12144
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md
modified_files:
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_download_limits.py
updated_date: 2026-07-18 21:49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refresh PR #2613 onto latest dev, migrate its colliding Backlog task identity, verify the oversized audio download regression executes the downloader with the required request contract and fails before body consumption, and reconcile all review threads and fresh CI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The oversized Content-Length regression invokes download_audio_file and asserts AudioFileSizeError.
- [x] #2 The test verifies one streamed request to the intended URL, no response-body iteration, and no target file creation.
- [x] #3 No production behavior is changed by this test-only audit remediation.
- [ ] #4 All PR review threads and latest-dev integration effects are reconciled.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: migrate the colliding task record and rebase onto current origin/dev. Stage 2: review the latest-dev diff and reproduce any remaining test defect. Stage 3: make only confirmed test or tracking fixes. Stage 4: run focused verification, independent review, push, resolve threads, and record fresh CI.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased onto origin/dev 668b0fce5707134768f880b5d064ccc5b0cc4691. The original active-invocation audit fix is already present in latest dev, so the final PR delta was reviewed as incremental contract coverage. Replaced the brittle exact headers/timeout equality with assertions for one request to the intended URL, stream=True, and zero iter_content calls, proving oversized Content-Length is rejected before response-body consumption. The shared fake response now uses yield from, clearing the touched-file Ruff diagnostic.

Verification: 3 focused tests pass with 14 warnings; Ruff passes the touched test file; git diff --check passes. Bandit reports 7 LOW B101 findings, all expected pytest assert statements in the test-only file, with 0 errors and no production scope. Independent specification and code-quality/security re-reviews approved with no remaining actionable findings.

Tracking: the PR's old TASK-12144 was archived before rebase because latest dev contains unrelated records with that ID. TASK-13001 is the active authoritative record.
Final pre-push latest-dev check: merge-base equals origin/dev 29acaca8c781213e27b12066372df13855e2e7a6. Reverification on this base: 3 tests passed with 14 warnings, Ruff passed, Bandit remained 7 LOW B101 pytest-assert findings with 0 errors, and diff checks passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused audio download limit tests pass on latest dev.
- [x] #2 Applicable Ruff checks pass or unrelated diagnostics are documented.
- [x] #3 Bandit findings are limited to expected pytest assert usage because no production code changes.
- [x] #4 git diff --check passes.
- [x] #5 Independent specification and code-quality reviews have no unresolved actionable findings.
- [ ] #6 PR review threads and fresh CI state are reconciled.
<!-- DOD:END -->
