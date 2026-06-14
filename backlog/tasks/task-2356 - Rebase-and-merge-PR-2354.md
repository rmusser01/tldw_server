---
id: TASK-2356
title: Rebase and merge PR 2354
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-14 19:08'
labels:
  - pr
  - review-remediation
  - media-multi
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track the review-comment remediation, latest dev rebase, verification, push, and merge for GitHub PR #2354 (`uat/media-multi-review`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PR #2354 is rebased on latest origin/dev without conflicts.
- [ ] #2 All open PR review comments are addressed in code or documented with rationale.
- [ ] #3 Focused frontend/unit/e2e verification passes or any blocker is documented.
- [ ] #4 PR branch is pushed and PR is merged into dev.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review-comment remediation:
- Removed the hardcoded fallback `TLDW_API_KEY` from `apps/tldw-frontend/scripts/media-multi-uat-driver.mjs`; the UAT driver now requires an explicit env API key.
- Updated the UAT driver from stale `/media` selectors (`results-select-*`, `media-bulk-*`) to `/media-multi` selectors (`media-review-result-row`, `media-multi-batch-*`).
- Replaced hardcoded `/tmp` artifact paths with `os.tmpdir()` and `path.join(...)`.
- Deferred `listVirtualizer.measureElement(el)` with `requestAnimationFrame`, with a disconnected-node guard and `setTimeout` fallback, to avoid the React flushSync-in-lifecycle warning path.
- Added a focused source-level regression contract for these review comments.

Verification before rebase/commit:
- `node --check scripts/media-multi-uat-driver.mjs` passed.
- `./node_modules/.bin/vitest run __tests__/media-multi-review-contracts.test.ts` passed (3 tests).
- `./node_modules/.bin/eslint __tests__/media-multi-review-contracts.test.ts scripts/media-multi-uat-driver.mjs` passed.
- `git diff --check` passed.
- UI regression tests passed: `MediaReviewPage.stage5.batch-toolbar.test.tsx` and `MediaReviewPage.stage5.export-trash-handoff.test.tsx` (5 tests). The local worktree has a stale `antd` symlink, so it was temporarily pointed at the installed package and restored afterward.
- Focused Playwright regression passed: `e2e/media-multi-bulk-select.spec.ts --project=chromium` (2 tests) against the running backend.
- Bandit: no Python files touched (`git diff --name-only -- '*.py'` empty). A direct Bandit invocation on the JS/TS touched files produced parse errors and no findings, so there is no applicable Python Bandit scope.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
