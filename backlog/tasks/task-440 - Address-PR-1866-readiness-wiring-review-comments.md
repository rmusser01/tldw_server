---
id: TASK-440
title: Address PR 1866 readiness wiring review comments
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-19 19:22'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the second live PR #1866 review batch for Character Chat readiness wiring, including model catalog/send-blocked readiness inputs, server recovery settings routing, and robust test listener cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Verify Qodo and CodeRabbit comments against current code before patching.
- [x] #2 Playground passes chat model catalog and send-blocked state into Character Chat readiness.
- [x] #3 Server recovery action opens the unscoped server settings target instead of provider-scoped model settings.
- [x] #4 Playground cockpit listener tests clean up global listeners even if assertions fail.
- [x] #5 Focused frontend tests pass and verification results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified Qodo and CodeRabbit comments against current Playground wiring. Added Character Chat readiness catalog loading in Playground via fetchChatModels({ returnEmpty: true }), passed availableModels and send-blocked state into buildCharacterChatReadiness, split model settings and server settings recovery so server recovery dispatches an unscoped settings event, and wrapped global listener assertions in try/finally.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the second live PR #1866 review batch. Verification: red cockpit-shell tests failed for unavailable model, send-blocked readiness, and scoped server recovery before the fix; focused Vitest suite passed with 6 files and 90 tests; git diff --check passed. TypeScript still fails on existing repo-wide baseline debt, and /tmp/tldw_pr1866_review2_tsc.txt has no errors matching the files touched in this review pass.
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
