---
id: TASK-436
title: Add WebUI Workflow Editor status shell
status: Done
labels:
- webui
- extension
- ux-remediation
- routes
- wp10
- workflow-editor
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a compact status-first shell to the Workflow Editor that makes step-type availability, validation state, dirty/save state, import/export/run access, and mobile panel access visible without changing graph logic or backend APIs. Add route error boundary coverage for /workflow-editor if missing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workflow Editor exposes step-type availability and failures in the editor shell using existing store state.
- [x] #2 Validation state is visible and named even when no issue icon is shown, including issue counts when present.
- [x] #3 Save dirty state, import, export, and run controls remain discoverable from the shell.
- [x] #4 Mobile panel access remains reachable on non-desktop viewports.
- [x] #5 Standalone /workflow-editor route has route error boundary coverage consistent with other option routes.
- [x] #6 Focused Vitest, Playwright browser verification, and diff check are recorded; Bandit is not applicable unless Python is touched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a compact Workflow Editor status summary below the toolbar. It reports step-type registry state, step-type failures, validation state/counts, saved/unsaved state, and run state using existing store fields.
- Promoted Import, Export, and Run panel access into the visible shell while leaving the existing toolbar, More menu, canvas, graph logic, and backend API contract unchanged.
- Kept mobile panel access intact through the existing `Open workflow panels` control and made the Run summary action open the execution panel/drawer on non-desktop viewports.
- Wrapped the standalone `/workflow-editor` route in `RouteErrorBoundary` and added a route-shell regression test.
- Browser verification initially exposed a timing-sensitive More actions assertion. The dropdown behavior was still present; the test now waits for the actual `New Workflow` menu item rather than synchronously checking a boolean.
- Bandit was not run because this slice touched frontend TypeScript/TSX, Playwright coverage, and Backlog metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Workflow Editor now exposes the operational state users need before working in the canvas: step-type availability/failure, validation status, saved/unsaved state, run state, direct import/export actions, and direct run-panel access. The route now has route error-boundary coverage. Focused unit tests, route-shell tests, Playwright browser verification, and targeted diff checks passed.
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
