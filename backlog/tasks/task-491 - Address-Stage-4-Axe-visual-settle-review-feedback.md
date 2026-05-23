---
id: TASK-491
title: Address Stage 4 Axe visual settle review feedback
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-23 04:36
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix PR review feedback: Stage 4 high-risk Axe scans should use waitForVisualSettle before AxeBuilder.analyze and reduce fixed sleep reliance in the scan path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stage 4 Axe high-risk scans call waitForVisualSettle before AxeBuilder.analyze.
- [x] #2 Stage 4 Axe high-risk scans no longer use direct networkidle waits or the fixed 250ms sleep in route-settle flow.
- [x] #3 Focused guard tests and diff checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated Stage 4 Axe high-risk route scanning to use waitForVisualSettle(page, LOAD_TIMEOUT) before AxeBuilder.analyze and replaced the direct networkidle plus fixed 250ms route-settle delay with the visual-settle helper. Removed the unused duplicate Axe retry helper so the guarded retry path is the active scan path. Added a focused Vitest guard to prevent future drift. Verification: new guard was red before the fix and passes after; Stage 4 Axe helper plus visual-settle guard passed (5 tests); existing e2e harness Stage 4 smoke-slice guard passed by name; git diff --check passed. Bandit skipped because only TypeScript test/e2e files were touched.
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
