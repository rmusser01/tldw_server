---
id: TASK-2401
title: Implement flashcards UX PR 0 evidence and harness refresh
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-23 17:21'
labels:
  - ux
  - flashcards
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refresh the Flashcards UX evidence and test harness before behavior remediation work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-23-flashcards-remaining-ux-remediation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented PR 0 flashcards UX evidence and harness refresh.

Changed:
- Added deterministic E2E seed IDs and prefix-safe cleanup helpers.
- Normalized E2E backend URLs so the documented TLDW_E2E_SERVER_URL=127.0.0.1:8000 command resolves to http://127.0.0.1:8000.
- Added authenticated flashcards API preflight and skip gating for backend-dependent Playwright cases.
- Added current-state component coverage for zero-deck IA, invalid import failure recovery, visible review re-rate, and create-drawer deck-reference lazy loading.

Verification:
- git diff --check: passed.
- URL normalization check: passed and printed http://127.0.0.1:8000.
- Focused Vitest command: 4 files passed, 91 tests passed, 1 skipped.
- Playwright flashcards workflow: 9 passed, 11 backend-dependent cases skipped when backend/flashcards preflight unavailable.
- Spec compliance review: approved.
- Code quality review: approved.

Bandit: not applicable; this PR slice touched frontend/E2E TypeScript and Backlog metadata only.

Notes:
- The clean PR worktree already had an unrelated TASK-2354 from origin/dev, so TASK-2401 is the non-conflicting worktree Backlog record committed with this slice.
- The worktree dependency layout has a broken tracked antd symlink target; verification temporarily repointed it to an installed cache target and restored it before staging.

Review follow-up:
- Rebased PR #2464 onto latest `origin/dev`.
- Removed the skipped future invalid-import inline alert test from PR0 instead of carrying disabled coverage.
- Replaced random/time-based E2E run IDs with deterministic IDs derived from Playwright test metadata plus a stable hash suffix.
- Scoped cleanup to the exact per-test run ID and made teardown cleanup best-effort with warnings instead of thrown failures.
- Fixed duplicate DESCRIPTION section markers in this Backlog task.

Review follow-up verification:
- Targeted grep for skipped tests, random/time IDs, no-arg seed calls, and old cleanup helpers: no matches.
- git diff --check: passed.
- Focused Vitest flashcards suite: 4 files passed, 91 tests passed.
- Playwright flashcards workflow: 9 passed, 11 backend-dependent cases skipped because the local backend preflight was unavailable; cleanup warnings did not fail teardown.
- Bandit: not applicable; review follow-up touched frontend/E2E TypeScript and Backlog metadata only.
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
