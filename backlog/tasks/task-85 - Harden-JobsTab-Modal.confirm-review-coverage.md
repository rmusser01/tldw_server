---
id: TASK-85
title: Harden JobsTab Modal.confirm review coverage
status: Done
assignee: []
created_date: '2026-05-05 19:29'
updated_date: '2026-05-05 19:32'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address follow-up review feedback for JobsTab delete confirmation tests by asserting Modal.confirm directly and adding cancel-path coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 JobsTab undo-delete test imports Modal and asserts Modal.confirm was called once before invoking onOk
- [x] #2 The test retrieves the confirmation config from Modal.confirm.mock.calls and explicitly invokes onOk for delete flows
- [x] #3 A cancel-path regression invokes onCancel if present and asserts delete/remove/undo side effects do not run
- [x] #4 Focused JobsTab undo-delete Vitest file passes
- [x] #5 git diff --check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Patch only JobsTab.undo-delete.test.tsx: replace the mirror modalConfirmMock with direct Modal.confirm assertions, keep production JobsTab onOk gating unchanged, add cancel-path regression coverage, then run focused Vitest and diff-check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented test-only Modal.confirm hardening. The antd mock now stores the real Modal.confirm call without a mirror helper, getPendingDeleteConfirmation asserts Modal.confirm directly and reads config from Modal.confirm.mock.calls, and a cancel-path regression confirms delete/remove/undo side effects do not run unless onOk is invoked. Verification: focused JobsTab undo-delete Vitest passed 1 file / 4 tests.

Verification: git diff --check passed. Bandit is not applicable because the touched source is a frontend Vitest test plus Backlog task metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the JobsTab delete confirmation test to model the antd Modal.confirm flow directly. The mock now stores the config without executing onOk, the helper asserts Modal.confirm was called once and reads the config from Modal.confirm.mock.calls, and the delete tests explicitly invoke onOk before expecting destructive side effects. Added cancellation coverage to prove cancel/no confirmation does not call deleteWatchlistJob, removeJob, or showUndoNotification. Verification passed: focused JobsTab undo-delete Vitest file and git diff --check. Bandit skipped as not applicable for frontend test-only changes.
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
