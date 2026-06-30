---
id: TASK-12043
title: Adopt admin sandbox diagnostics recovery state
status: Done
created_date: 2026-06-26 06:31
labels:
- webui
- capability-state
- admin
references:
- TASK-420
- TASK-418.10.4
- TASK-12042
documentation:
- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
modified_files:
- Docs/superpowers/plans/2026-06-26-webui-stage14-admin-sandbox-diagnostics-recovery-plan.md
- apps/packages/ui/src/components/Option/Admin/MonitoringDashboardPage.tsx
- apps/packages/ui/src/components/Option/Admin/__tests__/MonitoringDashboardPage.test.tsx
updated_date: 2026-06-26 06:37
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the deferred WebUI capability/error-state follow-up for the Admin monitoring sandbox runtime diagnostics failure state. Replace the generic diagnostics error alert with the shared RecoveryCallout and structured, non-secret endpoint diagnostics while preserving the existing monitoring dashboard layout and retry behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Forbidden sandbox diagnostics failures show user-language access-denied copy in the shared RecoveryCallout.
- [x] #2 Sandbox diagnostics raw endpoint/status/error details are available only as structured diagnostics, not the primary message.
- [x] #3 Existing Admin monitoring dashboard success, empty, and guard states continue to render.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
['Create a stage-specific plan document for this narrow route slice.', 'Add a focused failing MonitoringDashboardPage regression for the forbidden sandbox diagnostics failure state.', 'Implement the minimal RecoveryCallout/buildCapabilityState conversion for sandbox diagnostics errors.', 'Run the focused Admin monitoring tests, lint touched TS/TSX files, and diff checks.']
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD/verification notes:
- RED: `bun run test:run ../packages/ui/src/components/Option/Admin/__tests__/MonitoringDashboardPage.test.tsx -t "distinguishes forbidden sandbox diagnostics from unavailable diagnostics"` failed because the sandbox diagnostics access-denied title was not inside `RecoveryCallout`.
- GREEN: targeted regression passed after converting the sandbox diagnostics error state to `buildCapabilityState` + `RecoveryCallout`.
- GREEN: full `MonitoringDashboardPage.test.tsx` suite passed: 15 tests.
- Lint: direct ESLint on `MonitoringDashboardPage.tsx` and its test exited 0; only the known Next pages-directory notice was printed.
- Whitespace: `git diff --check` passed.
- Design-state verifier: `bun run verify:design-system-state` remains blocked by local `ERR_MODULE_NOT_FOUND: Cannot find package 'typescript'` from `apps/packages/ui/scripts/design-system-product-state-rules.mjs`.
- Bandit: not applicable; this slice touched TS/TSX and Markdown only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted the Admin monitoring sandbox diagnostics failure state from a generic alert to the shared capability recovery pattern. The page now builds a `RecoveryCallout` with stable user-language copy, a retry action, and structured diagnostics containing the GET path, status, and raw error for operators. Also tightened local Admin dashboard row/error types while preserving the existing success, empty, host-local warning, and guard states.
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
