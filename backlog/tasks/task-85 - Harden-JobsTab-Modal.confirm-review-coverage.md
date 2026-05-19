---
id: TASK-85
title: Harden JobsTab Modal.confirm review coverage
status: Done
assignee: []
created_date: '2026-05-05 19:29'
updated_date: '2026-05-05 20:40'
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
- [x] #1 JobsTab undo-delete test imports Modal and asserts Modal.confirm opened before invoking onOk
- [x] #2 The test retrieves the latest confirmation config from vi.mocked(Modal.confirm).mock.calls and explicitly invokes onOk for delete flows
- [x] #3 A cancel-path regression requires onCancel and asserts delete/remove/undo side effects do not run
- [x] #4 Focused JobsTab undo-delete Vitest file passes
- [x] #5 git diff --check passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Replace the mirror modalConfirmMock with direct Modal.confirm assertions, add an explicit JobsTab onCancel handler so cancel coverage exercises a real callback, and run focused Vitest, tsc, and diff-check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented test-only Modal.confirm hardening. The antd mock now stores the real Modal.confirm call without a mirror helper, getPendingDeleteConfirmation asserts Modal.confirm directly and reads config from Modal.confirm.mock.calls, and a cancel-path regression confirms delete/remove/undo side effects do not run unless onOk is invoked. Verification: focused JobsTab undo-delete Vitest passed 1 file / 4 tests.

Verification: git diff --check passed. Bandit is not applicable because the touched source is a frontend Vitest test plus Backlog task metadata.

PR #1326 review pass: Gemini requested vi.mocked(Modal.confirm) instead of the double cast and removal of the Mock import. Qodo correctly noted the cancel regression was no-op because JobsTab did not define onCancel. Plan: first tighten the test to require a real onCancel function and use vi.mocked, verify the focused test fails before production change, then add an explicit no-op onCancel to JobsTab Modal.confirm and rerun focused verification/diff-check.

PR #1326 review fixes implemented. Red check: focused JobsTab undo-delete Vitest failed because confirmConfig.onCancel was undefined. Green check: added explicit no-op onCancel to JobsTab Modal.confirm, replaced the double Mock cast with vi.mocked(Modal.confirm), removed the unused Mock import, and focused Vitest passed 1 file / 4 tests.

Fresh verification before push: focused JobsTab undo-delete Vitest passed 1 file / 4 tests; frontend tsc --noEmit passed; git diff --check against the PR branch passed.

Rebased PR #1326 onto dev after PR #1325 merged. Resolved the overlapping JobsTab test changes into a single Modal.confirm helper using vi.mocked, kept the explicit onCancel production callback, and removed the redundant test-only "confirmation is not accepted" variant from the merged overlap.

Post-rebase verification: rebased PR #1326 onto current dev after PR #1325 merged; focused JobsTab undo-delete Vitest passed 1 file / 4 tests; frontend tsc --noEmit passed; git diff --check against origin/dev..HEAD passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1326 review feedback. JobsTab now provides an explicit no-op Modal.confirm onCancel handler so the cancel regression exercises a real callback rather than optional-chaining a missing property. The test asserts onCancel is present, invokes it, and verifies deleteWatchlistJob, removeJob, and showUndoNotification remain untouched unless onOk is invoked. Also replaced the brittle Modal.confirm double cast with vi.mocked(Modal.confirm) and removed the unused Mock import. Verification: focused JobsTab undo-delete Vitest passed 1 file / 4 tests, frontend tsc --noEmit passed, and git diff --check passed. Bandit is not applicable for this frontend React/test slice.

Post-rebase verification: PR #1326 now layers explicit JobsTab onCancel coverage on top of current dev. Focused JobsTab undo-delete Vitest passed 1 file / 4 tests, frontend tsc --noEmit passed, and git diff --check passed.
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
