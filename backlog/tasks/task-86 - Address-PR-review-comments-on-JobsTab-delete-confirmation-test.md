---
id: TASK-86
title: Address PR review comments on JobsTab delete confirmation test
status: Done
assignee: []
created_date: '2026-05-05 19:53'
updated_date: '2026-05-05 19:53'
labels:
  - frontend
  - watchlists
  - tests
  - review-fix
dependencies: []
documentation:
  - >-
    apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.undo-delete.test.tsx
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up on PR #1325 review comments by clarifying the no-confirmation test behavior and making the Modal.confirm helper resilient to multiple calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The confirmation helper returns the latest Modal.confirm call config without requiring exactly one call.
- [x] #2 The no-confirmation test name and assertions do not imply an onCancel handler exists when production does not provide one.
- [x] #3 Focused JobsTab undo-delete Vitest coverage passes after the review fix.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented PR #1325 review fix in the branch worktree: getDeleteConfirmConfig now waits for any Modal.confirm call and returns the most recent config. The no-acceptance test no longer calls a nonexistent onCancel handler; it asserts onCancel is undefined and verifies no delete side effects occur until confirmation is accepted.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1325 review comments on JobsTab undo-delete coverage. The Modal.confirm helper now returns the latest confirm config without assuming exactly one call. The no-confirmation test has been renamed to avoid implying production provides onCancel, asserts onCancel is undefined, and still verifies deleteWatchlistJob, removeJob, and undo notification side effects do not run unless onOk is invoked. Verification: focused JobsTab undo-delete Vitest file passes. Bandit is not applicable because this is frontend test-only TypeScript/React code.
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
