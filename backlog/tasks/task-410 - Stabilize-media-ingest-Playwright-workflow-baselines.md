---
id: TASK-410
title: Stabilize media ingest Playwright workflow baselines
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-17 03:29'
labels:
  - bulk-conference-ingest
  - e2e
  - qa
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the remaining broader media-ingest Playwright workflow failures observed after Task 9 so the full workflow file can be used as a reliable closeout gate. Scope is limited to current UI/test alignment for media page readiness, Quick Ingest trigger discovery, and legacy review-route navigation stability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Media ingest workflow tests stub offline media list/search reads so /media does not throw a Next runtime overlay without a live backend.
- [x] #2 Quick Ingest visible-trigger test accepts the current first-ingest Ingest CTA while preserving the legacy Open Quick Ingest fallback.
- [x] #3 Legacy review-route navigation test avoids the hydration-unstable pointer click while still verifying navigation to /media-multi.
- [x] #4 Focused and full-file Playwright verification is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Patched apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts only. Added empty media list/search route stubs in beforeEach for offline /media rendering, widened the visible Quick Ingest trigger expectation to include the current first-ingest CTA, and switched the legacy /review -> /media-multi navigation assertion to invoke the verified moved-route anchor from page context.

Verification recorded: focused baseline set passed (3 passed), conference playlist/extension handoff set passed (2 passed), legacy redirect focused test passed (1 passed), and full media-ingest file completed with 18 passed / 12 skipped. The skipped tests are existing backend-dependent cases when the backend health fixture is unavailable. Bandit was not run because this is a frontend Playwright test-only change with no Python/runtime code touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stabilized the media ingest Playwright workflow baselines for the browser/offline path and the bulk conference workflow. Full-file verification now completes with the current backend-dependent skips rather than failing on /media runtime overlays or the legacy route click race.
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
