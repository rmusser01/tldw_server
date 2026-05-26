---
id: TASK-437
title: Add WebUI Skills capability states
status: Done
labels:
- webui
- extension
- ux-remediation
- routes
- wp10
- skills
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Skills route capability-aware with clear loading, unsupported, no-connection, and supported states using existing shared UI/state patterns. Keep SkillsManager hidden until capability is known and add route error boundary coverage for /skills if missing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Skills loading state uses route-appropriate capability language and does not render SkillsManager before capability is known.
- [x] #2 Unsupported Skills API state uses existing shared state primitives and includes a recovery/update hint or action.
- [x] #3 No-connection/setup state remains clear and route-specific through WorkspaceConnectionGate behavior.
- [x] #4 Supported state renders SkillsManager only after capability is known.
- [x] #5 Standalone /skills route has route error boundary coverage consistent with other option routes.
- [x] #6 Focused Vitest, Playwright browser verification, and diff check are recorded; Bandit is not applicable unless Python is touched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Replaced the generic Skills capability skeleton with a named `StatePanel` loading state so users see that Skills API support is being checked and the manager is not mounted prematurely.
- Replaced the unsupported Skills empty state with a `RecoveryCallout` backed by `buildCapabilityState`, including `/api/v1/skills` diagnostics and a `Refresh capabilities` recovery action.
- Preserved the existing `WorkspaceConnectionGate` setup/no-connection behavior with route-specific Skills copy.
- Wrapped the standalone `/skills` route in `RouteErrorBoundary`.
- Updated the existing Skills Playwright spec copy matchers for the new unsupported heading and message.
- Existing backend-dependent Skills Playwright spec exited cleanly but skipped all tests because the configured backend was unavailable; direct browser QA against the local route with mocked no-Skills capability confirmed the unsupported state, diagnostics, and refresh action render.
- Bandit was not run because this slice touched frontend TypeScript/TSX, Playwright coverage, and Backlog metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Skills now has capability-aware loading, unsupported, setup/no-connection, and supported states. The Skills manager only renders once support is known. Unsupported servers now get a shared recovery callout with diagnostics and refresh action, and `/skills` has route error-boundary coverage.
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
