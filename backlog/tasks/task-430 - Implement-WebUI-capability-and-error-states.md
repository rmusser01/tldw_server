---
id: TASK-430
title: Implement WebUI capability and error states
status: Done
labels:
- ux
- webui
- extension
- states
- implementation
priority: high
references:
- TASK-420
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
- Docs/Reviews/WEBUI_EXTENSION_UX_HCI_AUDIT_2026_05_17.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved WebUI/extension capability and error-state remediation slice from TASK-420. Scope is shared capability-state expectations plus first adopter routes `/sources`, `/scheduled-tasks`, and `/integrations`; raw endpoint/status details must move behind diagnostics instead of being the primary page message. No backend API changes or broad redesign in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared state primitive expectations cover visible state labels, user-language messages, recovery actions, and diagnostics isolation for raw endpoint details.
- [x] #2 A pure capability-state mapping helper is added only if at least two first-adopter routes would duplicate error-to-state mapping.
- [x] #3 `/sources` uses user-language unavailable/setup/auth/permission/network states with raw endpoint/status details only in diagnostics.
- [x] #4 `/scheduled-tasks` uses user-language unavailable/setup/auth/permission/network states with raw endpoint/status details only in diagnostics.
- [x] #5 `/integrations` uses user-language unavailable/setup/auth/permission/network states with raw endpoint/status details only in diagnostics.
- [x] #6 Focused Vitest coverage passes for shared state behavior and changed route states.
- [x] #7 Browser QA is run for changed pages when the frontend/backend can be started; any environment blocker is documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `buildCapabilityState()` and related helpers under the shared state primitives so first-adopter routes use the same unavailable, setup-required, auth-required, permission-denied, degraded, and network-state vocabulary.
- Extended `RecoveryCallout` to render permission-denied states and diagnostics-only raw details.
- Updated `/sources`, `/scheduled-tasks`, and `/integrations` so endpoint paths, status codes, server URLs, and raw response text are diagnostics content rather than the primary page message.
- Preserved existing route/page ownership and backend API contracts. No backend code was changed.
- Touched frontend files:
  - `apps/packages/ui/src/components/ui/state/capability-state.ts`
  - `apps/packages/ui/src/components/ui/state/index.ts`
  - `apps/packages/ui/src/components/ui/state/RecoveryCallout.tsx`
  - `apps/packages/ui/src/components/Option/Sources/SourcesAvailabilityGate.tsx`
  - `apps/packages/ui/src/components/Option/Sources/SourcesWorkspacePage.tsx`
  - `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
  - `apps/packages/ui/src/components/Option/Integrations/IntegrationManagementPage.tsx`
  - `apps/packages/ui/src/components/Option/Integrations/IntegrationPolicyPanel.tsx`
  - focused tests under the matching `__tests__` directories and route wrapper tests.
- Browser QA observations:
  - Frontend dev server started on `http://127.0.0.1:18031`.
  - Initial route QA without backend showed the app-level server readiness gate, not route content.
  - Backend startup inside the sandbox failed with `[Errno 1] error while attempting to bind on address ('127.0.0.1', 8000): operation not permitted`; escalated startup succeeded.
  - With backend running, browser-rendered `/sources` showed `Sources are unavailable on this server` and kept `GET /api/v1/ingestion-sources`, status `404`, and raw `Not Found` text in Diagnostics.
  - With backend running, browser-rendered `/scheduled-tasks` showed `Scheduled tasks are unavailable on this server` and kept `GET /api/v1/scheduled-tasks`, status `404`, and raw `Not Found` text in Diagnostics.
  - With backend running, browser-rendered `/integrations` showed `Personal integrations are unavailable on this server` and kept `GET /api/v1/integrations/personal` in Diagnostics.
  - Browser screenshots were captured under `/tmp/tldw-task430-qa/`.
- Verification:
  - `bunx vitest run src/components/ui/state/__tests__/capability-state.test.ts src/components/ui/state/__tests__/state-primitives.test.tsx src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/components/Option/Integrations/__tests__/IntegrationManagementPage.test.tsx src/routes/__tests__/option-sources-route-guards.test.tsx src/routes/__tests__/scheduled-tasks-route.test.tsx src/routes/__tests__/integrations-route.test.tsx` passed with 8 files and 50 tests.
  - `git diff --check` passed.
  - Full `tsc --noEmit` is not clean in the current package baseline. The pinned local TypeScript run emitted existing package-wide errors; grep of `/tmp/tldw_ui_tsc_task430.log` found no touched-file matches for this slice.
  - `bun run verify:design-system-state` is blocked in this checkout because the verifier cannot resolve the `typescript` package from the script runtime.
  - Bandit is not applicable because no Python code was touched.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the first capability/error-state remediation slice for `/sources`, `/scheduled-tasks`, and `/integrations`. The changed pages now use shared user-language states with raw endpoint/status details hidden behind diagnostics, backed by focused component/route tests and browser observations against the running WebUI plus local API.

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Focused tests or verification recorded
- [x] #3 Documentation or Backlog notes updated when relevant
- [x] #4 Bandit run for touched Python code when applicable or documented as not applicable
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
