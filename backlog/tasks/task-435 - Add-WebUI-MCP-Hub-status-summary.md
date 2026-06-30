---
id: TASK-435
title: Add WebUI MCP Hub status summary
status: Done
labels:
- webui
- extension
- ux-remediation
- routes
- wp10
- mcp
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a compact status-first summary to the MCP Hub using existing workflow/view metadata, keep workflow controls and FTUX behavior intact, and add route error boundary coverage. This is frontend-only and must not add backend API calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP Hub shows a compact status summary for servers, credentials, policy assignments, approvals, workspaces, and audit before deep tab detail.
- [x] #2 Status summary actions navigate to existing MCP Hub workflow/view states without hiding current workflow controls.
- [x] #3 First-time explainer remains dismissible and diagnostics/details stay inside workflow/tab areas.
- [x] #4 The standalone /mcp-hub route has a route error boundary consistent with other option routes.
- [x] #5 Focused Vitest, Playwright browser verification, and diff check are recorded; Bandit is not applicable unless Python is touched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a compact MCP Hub status summary in `McpHubPage` before the FTUX explainer and workflow controls. The cards cover servers/credentials, policy assignments, approvals, workspace boundaries, and audit findings, and each action reuses existing workflow/view query-state navigation.
- Kept diagnostics and detail surfaces inside the existing workflow tabs. The first-time explainer remains dismissible and still persists through the existing FTUX storage helper.
- Wrapped the standalone `option-mcp-hub` route with `RouteErrorBoundary` using the route id and label expected by other option routes.
- Added focused page, FTUX, route shell, and browser assertions for the summary and status actions.
- Bandit was not run because this slice touched frontend TypeScript/TSX and Backlog metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
MCP Hub now has a status-first summary above the deep workflow area, with summary actions that open the existing workflow/view states rather than introducing a separate navigation model. The `/mcp-hub` option route now has route error-boundary coverage. Verification passed with focused Vitest, Playwright browser QA for the MCP Hub route, and targeted whitespace checks.
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
