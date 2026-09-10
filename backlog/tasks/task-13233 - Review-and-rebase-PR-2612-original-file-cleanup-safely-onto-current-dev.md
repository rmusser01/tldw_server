---
id: TASK-13233
title: Review and rebase PR 2612 original-file cleanup safely onto current dev
status: In Progress
assignee: []
created_date: '2026-09-10 00:49'
updated_date: '2026-09-10 02:29'
labels:
  - media
  - storage
  - pr-followup
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2612'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refresh PR #2612 onto current dev, verify continued applicability, and resolve shared-original deletion, stale retrieval, cancellation, and test-isolation issues. Supersedes this PR media-cleanup record TASK-13000, whose ID is now owned by an unrelated UserProfiles task on dev; preserve that prior record in archive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rebase retains useful compensation for original-file registration failures on current dev.
- [x] #2 Cleanup exceptions preserve the original failure and cancellation propagates.
- [x] #3 Focused and adjacent tests, no-new-lint comparison, Bandit, pre-commit checks, and independent review pass.
- [x] #4 Existing PR branch is updated with a lease-protected push; current CI and human Change summary merge gates are reported.
- [x] #5 Failed registration preserves the previous original; successful replacement selects the latest original and safely retires superseded binaries while preserving plaintext history.
- [ ] #6 Latest dev rebase, all current Qodo feedback, and required CI are resolved before merging the user-authorized PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: verify current-dev applicability and rebase in an isolated worktree. Stage 2: reproduce review findings, implement minimal safe cleanup and retrieval fixes, and run focused/adjacent verification plus independent review. Stage 3: publish the existing PR branch with an explicit force-with-lease and report current CI and the human Change summary gate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased original head b16d65a33407682425c9dd1539a63aea6ea37a5e onto origin/dev 40345571a2cfc8b3a8893545836097d27e4ee86c without conflicts; a final fetch confirmed dev unchanged. All four original regression cases fail against dev because no compensating delete occurs. New real SQLite/filesystem tests reproduce shared-path deletion, overwrite damage, and stale retrieval; cancellation regression reproduces swallowed CancelledError. Fixed with per-attempt UUID filenames, newest matching MediaFiles registration selection, and cancellation propagation through the three enclosing handlers. Fixed quota singleton leakage in unit fixtures and five newly introduced Ruff typing diagnostics. Final validation: 134 tests passed, 8 warnings; production Bandit 0 findings and 0 errors over 5943 LOC; compileall and git diff --check pass; applicable pre-commit checks pass. Whole-file Ruff diagnostics match dev exactly: persistence 25, media_files_repository 2, original-storage tests 21, MediaFiles tests 1; no new diagnostics. Independent final code/security review approved. Both historical GitHub review threads are resolved. Earlier PR task IDs collided with dev; media TASK-13000 and transient child are archived, unrelated dev task files unchanged. Retention and quota-accounting follow-up is TASK-13233.1: earlier successful blobs remain registered and retained, with no automatic expiry guarantee. The final pushed head still requires GitHub CI and a human-written Change summary before merge; local tests do not substitute for those gates.

User clarified latest-only binaries; the prior retention decision is superseded by TASK-13233.1. Implemented retirement after committed replacement while preserving all plaintext history. Independent review approves; 153 tests pass and Bandit remains clean. Rebased again without conflicts onto dev 456eafb7a603449722ba8db806071a5e2aa5e7d6 (only intervening VZ Go changes); post-rebase validation in progress. The separate pre-existing quota-root concern is preserved in TASK-13233.2.

Final post-rebase validation also passed: 153 tests, 8 existing warnings, 74.50 seconds. Range-diff shows all seven commits unchanged by the final rebase. Latest-only original replacement TASK-13233.1 is complete; old binary retention is no longer the normal successful-upload policy.

Requester explicitly authorized rebasing onto latest dev, addressing all issues/comments after Qodo posts, and merging. Their human-written Change summary is published verbatim. Starting from local c39a1a798d; remote PR is now ready for review at d0806eeeea with Qodo review pending. Preserve and inspect the intervening remote update before rebasing; use an explicit push lease and final-head merge guard.

Rebased all nine PR commits onto dev f0248aaa00047d2ffcc3bde295d9fbb8296add8a without conflicts. Range-diff confirms unchanged commits; code tree matches GitHub merge head d0806eeeea exactly, apart from this task record. Fresh targeted/adjacent verification including latest-dev audio download regression: 156 passed, 8 existing warnings, 83.21 seconds. Bandit 0 findings/0 errors across four production files. Qodo review started at 2026-09-10T01:51:12Z and remains pending. Publish with explicit lease against d0806eeeeab8d0b0bf281a4715ff82f24989ab1f; address posted feedback and pass final-head required checks before authorized merge.

Qodo posted nine findings in issuecomment-5611521549. Validating each rather than applying speculative changes: fixed snapshot row IDs contradict the claimed all-originals deletion interleaving; StorageBackend.delete already specifies False for absent objects. Investigating shared paths and supported late path reuse. Planned corrections cover cross-media reference protection, off-event-loop database cleanup, traceback logging, explicit test annotations/docstrings, and a documented public cleanup operation with observable-behavior tests. Add interleaving regressions and preserve existing replacement semantics.

Qodo remediation reproduced and corrected shared references, slow synchronous DB work, traceback loss, and deterministic OpenWebUI path reuse; all new tests annotated/documented and cleanup exposed as a supported public operation. Disputed stale-cleaner data-loss claim and missing-blob retry claim have behavior/contract evidence. Follow-up review found shared-path retirement orphaning and UUID failed-attempt leaks; all three new regressions failed before correction, and focused replacement/hydration suite now passes 46 tests. Final adjacent suite and independent review pending.

Final Qodo correction validation: 190 targeted/adjacent tests passed with 8 existing warnings (81.43 seconds); Bandit 0 findings/0 errors over all five production files; Ruff no new diagnostics against dev; AST audit found no missing annotations/docstrings in added functions; Black, applicable pre-commit, syntax, and diff checks pass. Independent final review found no remaining actionable findings and recommends publishing. All nine Qodo dispositions are ready to post, including evidence-backed rebuttals of findings 1 and 9. Final-head CI and merge remain pending.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Qodo code corrections are complete and independently reviewed. All 190 targeted/adjacent tests pass, Bandit is clean, and no new lint diagnostics were introduced. Latest dev remains f0248aaa00047d2ffcc3bde295d9fbb8296add8a. Publishing corrections and resolving review threads precedes final-head CI and the already-authorized merge.
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
