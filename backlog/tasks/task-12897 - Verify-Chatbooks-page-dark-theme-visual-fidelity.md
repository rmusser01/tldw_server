---
id: TASK-12897
title: Verify Chatbooks page dark-theme visual fidelity
status: Done
assignee: []
created_date: '2026-07-05 20:43'
updated_date: '2026-07-05 22:31'
labels:
  - frontend
  - theme
  - webui
  - chatbooks
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Walk through the WebUI Chatbooks page in dark mode against a real backend, check menus/options for light-theme or low-contrast visual drift, apply the smallest shared-token/component fix if needed, and record focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chatbooks page dark-mode walkthrough has no large light-surface leaks.
- [x] #2 Chatbooks menus/options and primary controls are covered by rendered visual QA.
- [x] #3 Real backend requests used by the walkthrough are recorded with status results.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Real-backend walkthrough completed against frontend http://127.0.0.1:8081 and backend http://127.0.0.1:18001. Covered Chatbooks export, media quality select, import, source select, OpenWebUI JSON hydration controls, conflict select, and jobs. Captured screenshots under /private/tmp/tldw-real-chatbooks-dark. Report: /private/tmp/tldw-real-chatbooks-dark/report.json. Result: 0 light-surface leaks, 0 low-contrast text leaks, no request failures, Chatbooks export/import job endpoints returned 200.

Added Chatbooks coverage to apps/tldw-frontend/e2e/smoke/dark-theme-visual-fidelity.spec.ts for export, lower export pickers, export media select, import, import source select, OpenWebUI JSON hydration controls, conflict select, and jobs. Verification: npx playwright test e2e/smoke/dark-theme-visual-fidelity.spec.ts --reporter=line passed; git diff --check passed. Bandit skipped because touched repository files are frontend Playwright TypeScript and Backlog Markdown only; no Python code changed.

PR review follow-up: rebased on latest dev, aligned Chatbooks job and health mocks with backend response shapes, waited for Ant dropdowns to close before continuing, replaced hardcoded wheel scrolling with locator-based scrolling, and added aria labels to the Chatbooks media quality, import source, and conflict resolution selects so the smoke test can use semantic locators.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reviewed the Chatbooks page in dark mode against a real backend and found no light-theme visual drift in covered states. Added mocked Playwright dark-theme regression coverage for Chatbooks alongside the existing Chat/Notes/Characters smoke so the export/import/jobs surfaces and key dropdown menus remain guarded.
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
