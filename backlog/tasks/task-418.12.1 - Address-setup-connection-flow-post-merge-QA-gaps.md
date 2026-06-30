---
id: TASK-418.12.1
title: Address setup connection flow post-merge QA gaps
status: Done
labels:
- ux
- webui
- extension
- setup
- follow-up
priority: High
parent_task_id: TASK-418.12
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to the completed WebUI setup/connection-flow remediation after post-merge continuation. Scope: lock the connection UX route-state matrix around active health checks, keep setup/recovery browser QA aligned with the configured-server readiness branch, and verify the setup-adjacent route suite without backend/API changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Connection UX state matrix covers active health-check/testing states and connected/degraded/error/demo states.
- [x] #2 Production connection state derivation is changed only for the proved active-check gap.
- [x] #3 Setup/recovery Playwright QA passes for desktop and mobile, including the configured-server setup readiness branch.
- [x] #4 Focused package UI and frontend Vitest gates pass.
- [x] #5 Focused Playwright login, onboarding, hosted-placeholder, and setup-connection-flow gates pass or documented skips are recorded.
- [x] #6 No backend APIs, route renames, or broad visual redesign are included.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation notes:
- Added focused connection-store matrix coverage for deriveConnectionUxState, including health-check testing, SEARCHING, connected_ok, connected_degraded, auth/unreachable errors, and demo mode.
- Verified the red failure first: UNCONFIGURED + configStep health + isChecking mapped to unconfigured instead of testing.
- Reordered deriveConnectionUxState so demo and stable connected states remain first, then active SEARCHING/isChecking states map to testing before unconfigured fallback handling.
- Updated setup-connection-flow Playwright QA to accept both unconfigured Setup Wizard and configured-server Setup readiness branches while still requiring exactly one h1, no normal navigation chrome, and no horizontal overflow.
- No backend APIs, route renames, broad visual redesign, or unrelated cleanup were included.

Verification:
- bunx vitest run src/store/__tests__/connection.test.ts -> 16 tests passed after the red failure was fixed.
- bunx vitest run src/routes/__tests__/option-index.setup-flow.test.tsx -> 6 tests passed.
- bunx vitest run src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx -> 4 tests passed; existing react-i18next no-instance warning remains in this test path.
- bunx vitest run __tests__/navigation/route-placeholder-component.test.tsx __tests__/navigation/route-redirect-component.test.tsx __tests__/navigation/not-found-page.test.tsx -> 16 tests passed.
- bunx vitest run src/store/__tests__/connection.test.ts src/routes/__tests__/option-index.setup-flow.test.tsx src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx -> 26 tests passed.
- bunx playwright test e2e/workflows/setup-connection-flow.spec.ts --reporter=line -> first sandbox run failed with EPERM binding port 8080; escalated rerun exposed stale Setup Wizard-only browser expectation; after test patch, 4 tests passed.
- bunx playwright test e2e/login.spec.ts e2e/workflows/onboarding-ingestion-first.spec.ts e2e/workflows/hosted-placeholder-routes.spec.ts --reporter=line -> 12 tests passed.
- git diff --check -> passed.
- Bandit not applicable: touched scope is frontend TypeScript/TSX and Backlog markdown only; no Python files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the post-merge setup/connection-flow QA follow-up. The connection UX state matrix now locks active health checks as the testing route state, production derivation handles that proved gap without destabilizing demo or connected refresh states, and setup/recovery browser QA now matches both configured-server readiness and unconfigured setup branches. Focused package UI, frontend navigation, Playwright route, login, onboarding, and hosted-placeholder gates passed. No backend/API changes, route renames, or broad visual redesign were made.
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
