---
id: TASK-12105
title: Fix Quick Ingest YouTube defaults maximum update depth regression
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-10 17:00'
labels:
  - bug
  - frontend
  - quick-ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix the latest-dev Quick Ingest regression where adding a YouTube URL and clicking Use defaults & process triggers React's maximum update depth error. Include focused frontend regression coverage and browser verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Reproduce and identify the latest-dev Quick Ingest YouTube defaults render-loop path
- [x] Prevent `Use defaults & process` from entering processing when analysis requires a missing provider
- [x] Add focused unit and browser coverage for the regression path
- [x] Run targeted tests, typecheck, and browser verification
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Latest-dev investigation used a clean worktree at `.worktrees/investigate-quick-ingest-loop`. PR #2700 is present on `origin/dev` and `origin/main`; the original `/media` render-loop fixes remain.

Root cause: a later commit, `e204d26cce` (`Harden ingest analysis provider UX`), added an analysis-provider guard inside `QuickIngestWizardModal.startRun`. The default Quick Ingest preset has `perform_analysis=true` and no `api_name`, so clicking `Use defaults & process` moves the wizard into processing and then immediately creates a synthetic failed run. In a real browser that reproduced React `Maximum update depth exceeded` through the AntD/rc-portal modal path.

Fix in the clean worktree: detect the missing provider before `skipToProcessing` and keep the user on the add-content step with an inline warning; make the late guard reset back to the add step instead of synthesizing a failed run; shallow-stabilize the Quick Ingest session-store selectors used by the modal/widget; make repeated wizard progress/result reducer updates idempotent. Added a focused Vitest regression and a Playwright E2E spec for the YouTube defaults flow.

Verification: RED session test failed on latest behavior, then passed after fix (22/22); quick-ingest-batch service tests passed (33/33); frontend typecheck passed; Playwright browser regression passed after walking through the Quick Ingest YouTube defaults flow. Bandit was skipped because only frontend TypeScript and Backlog Markdown were touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the Quick Ingest YouTube defaults render-loop regression by validating the analysis provider before entering processing, keeping the user on the add-content step with an inline warning when defaults require analysis but no provider is selected. Added browser and unit regression coverage for the path and hardened the related modal/session update edges.

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
