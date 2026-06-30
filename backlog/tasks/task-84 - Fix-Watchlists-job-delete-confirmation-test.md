---
id: TASK-84
title: Fix Watchlists job delete confirmation test
status: Done
assignee:
  - codex
created_date: '2026-05-05 19:15'
updated_date: '2026-05-05 19:17'
labels:
  - frontend
  - watchlists
  - tests
dependencies: []
documentation:
  - >-
    apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.undo-delete.test.tsx
  - apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobsTab.tsx
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the JobsTab undo-delete test so it models the production Modal.confirm gate. The test should not delete immediately when the trash button is clicked; it should assert Modal.confirm is shown and explicitly invoke onOk to simulate user confirmation. Add cancel coverage if practical so a dismissed confirmation does not call deleteWatchlistJob.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The JobsTab undo-delete antd Modal.confirm mock stores confirmation config without auto-running onOk.
- [x] #2 Existing delete/undo test explicitly invokes the stored onOk callback before expecting deleteWatchlistJob and undo notification behavior.
- [x] #3 Cancel/dismiss coverage asserts invoking onCancel, when present, does not call deleteWatchlistJob.
- [x] #4 Focused JobsTab undo-delete Vitest coverage passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify the current JobsTab undo-delete test behavior in the clean worktree with the focused Vitest file.
2. Update only apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.undo-delete.test.tsx so the antd Modal.confirm mock records its config and never auto-runs onOk.
3. Add small test helpers to retrieve the latest confirm config, explicitly await config.onOk?.() in delete/undo flows, and add cancel coverage that proves deleteWatchlistJob is not called when the confirmation is dismissed.
4. Run focused Vitest for JobsTab.undo-delete.test.tsx, then run git diff --check and targeted ESLint if available. Document Bandit as not applicable because the change is frontend test-only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified the review target in origin/dev: JobsTab production deletion is gated through Modal.confirm with onOk executing executeDelete(job.id). The focused test baseline failed because the antd mock did not expose Modal, so clicking delete never reached a valid confirmation path.

Updated JobsTab.undo-delete.test.tsx only: added a Modal.confirm mock that records config without auto-confirming, added a helper to assert/retrieve the confirm config, explicitly invokes onOk in delete-path tests, and added cancel/dismiss coverage that does not call deleteWatchlistJob.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the Watchlists JobsTab undo-delete test so it models the production confirmation gate. The antd mock now includes Modal.confirm as a passive vi.fn that stores the config. Delete-path tests assert the confirmation was requested and explicitly invoke onOk before expecting deleteWatchlistJob, removeJob, and undo notification behavior. Added cancel/dismiss coverage that invokes onCancel when present and verifies no delete side effects occur.

Verification: focused JobsTab.undo-delete Vitest file passes with 4 tests. git diff --check passes. Targeted ESLint exits 0 with no errors; it reports pre-existing no-explicit-any warnings in this test mock. Bandit is not applicable because this is frontend test-only TypeScript/React code.
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
