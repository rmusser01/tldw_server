---
id: TASK-475
title: Investigate and stabilize existing UI test failures
status: Done
labels:
- tests
- ui
- workspaces
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track investigation of the existing frontend test failures surfaced while validating workspace-related UI changes. Scope includes reproducing failing tests, identifying root causes, applying narrow test/runtime fixes where necessary, and recording verification results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the deterministic existing failures found in the current test slices by updating the Markdown import, stabilizing test localStorage setup, preserving needed react-router-dom mock exports, giving Flashcards ReviewTab tests a default recent-sessions query shape, and correcting stale Admin/CreateTab test assertions. Fresh verification: UI targeted run passed 21 files / 111 tests; backend workspace API slice passed 51 tests / 6 warnings. Known limitation: the full apps/packages/ui package suite was not rerun cleanly after fixes because the earlier full run was started before fixes and was killed after becoming stale/noisy. Bandit not run because this task changed TypeScript/TSX test/setup code only, not Python application code.
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
