---
id: TASK-79
title: Refresh ReviewTab create CTA snapshots
status: Done
assignee: []
created_date: '2026-05-05 17:41'
updated_date: '2026-05-05 17:46'
labels:
  - frontend
  - tests
  - tldw-frontend
  - vitest
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the local tldw-frontend test stabilization series after PR #1316 merged. Use a fresh dev-based worktree and keep this slice narrow: reproduce and fix the ReviewTab.create-cta snapshot drift caused by the current review prompt-side segmented control appearing in the topbar.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh dev-based worktree records current tldw-frontend TypeScript baseline before edits
- [x] #2 Focused ReviewTab create CTA Vitest failure is reproduced before edits
- [x] #3 Root cause is documented from the snapshot diff and current ReviewTab markup
- [x] #4 Only the stale ReviewTab create CTA snapshots are updated
- [x] #5 Focused ReviewTab/RecentStudySessions Vitest command passes after the fix
- [x] #6 tldw-frontend TypeScript and git diff checks pass after the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run tldw-frontend TypeScript on the fresh origin/dev worktree to confirm the baseline remains green. 2. Run a focused RecentStudySessions/ReviewTab Vitest slice to identify the current failure. 3. Verify the snapshot diff against ReviewTab markup and confirm it reflects intentional UI state. 4. Update only the stale ReviewTab.create-cta snapshots. 5. Re-run the focused Vitest command, TypeScript, and git diff --check; record Bandit skip if only frontend tests changed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fresh worktree: /private/tmp/tldw-frontend-recent-study-gates on branch codex/tldw-frontend-recent-study-gates from origin/dev ef19f72f8, the PR #1316 merge commit. Dependency setup: bun install from apps completed. Baseline TypeScript passed with node node_modules/typescript/bin/tsc --noEmit --pretty false -p tsconfig.json. Focused Vitest command failed only in ReviewTab.create-cta snapshots: the received topbar includes the current flashcards-review-prompt-side-toggle segmented control with Front first / Back first options. RecentStudySessions and ReviewTab.study-suggestions passed in the same focused run.

Fix implemented: refreshed only the two stale ReviewTab.create-cta topbar snapshots. The diff adds the current flashcards-review-prompt-side-toggle segmented control to the active review and caught-up completion topbar snapshots. Verification after update: focused RecentStudySessions/ReviewTab Vitest command passed 3 files / 20 tests; tldw-frontend TypeScript passed; git diff --check passed. Bandit skipped because touched source is frontend TypeScript snapshot/test metadata only.

Opened PR #1318: https://github.com/rmusser01/tldw_server/pull/1318.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refreshed the next tldw-frontend Vitest slice after PR #1316 merged. The failing ReviewTab.create-cta snapshots were stale: ReviewTab now renders the prompt-side segmented control in the review topbar, so the active review and caught-up completion topbar snapshots needed to include Front first / Back first controls.

Verification: focused RecentStudySessions/ReviewTab Vitest passed 3 files / 20 tests; tldw-frontend TypeScript passed; git diff --check passed. Bandit was skipped because this slice only touched frontend snapshot/test task files. PR: https://github.com/rmusser01/tldw_server/pull/1318.
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
