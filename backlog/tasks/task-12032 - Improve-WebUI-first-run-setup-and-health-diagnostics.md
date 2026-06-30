---
id: TASK-12032
title: Improve WebUI first-run setup and health diagnostics
status: Done
assignee: []
created_date: 2026-06-25 22:04
labels:
- webui
- setup
- health
- ux
- onboarding
dependencies: []
priority: high
documentation:
- Docs/superpowers/plans/2026-06-25-webui-stage3-setup-health-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-25-webui-stage3-setup-health-plan.md
- apps/packages/ui/src/routes/option-setup.tsx
- apps/packages/ui/src/routes/__tests__/option-setup-readiness.test.tsx
- apps/packages/ui/src/components/Option/Settings/tldw-connection-status.ts
- apps/packages/ui/src/components/Option/Settings/__tests__/tldw-connection-status.test.ts
- apps/packages/ui/src/components/Option/Settings/health-status.tsx
- apps/packages/ui/src/components/Option/Settings/__tests__/health-status.design-system.test.tsx
- apps/packages/ui/src/components/Option/Settings/health-summary.tsx
updated_date: 2026-06-25 22:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 3 of the WebUI audit remediation roadmap: new self-host users need a guided connection path, health diagnostics must distinguish missing credentials from server outage, and onboarding should guide rather than block route exploration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /setup exposes a self-host connection path with server URL, API key, test connection, key-location help, and skip/explore affordance.
- [x] #2 Health diagnostics classify missing server URL, missing API key, invalid key, unreachable server, and degraded feature checks separately.
- [x] #3 Missing credentials no longer appear as a generic core-health server outage.
- [x] #4 Fresh-user onboarding guidance does not globally block chat/shell exploration when connection setup is the only missing item.
- [x] #5 Focused unit or E2E coverage records the setup and health behavior changed or verified.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 3 task created after Stage 1 commit 636853baea and Stage 2 commit 6db1b4c4ef. Planned areas: option-setup route/tests, health-status/connection-status/health-summary helpers, milestone/onboarding overlay triggers, focused onboarding/setup tests after investigation.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Stage 3 setup/health UX. /setup now has a self-host connection panel with server URL, password-masked API key, key-location help, Test connection, and Skip and explore UI. Skip writes assistant_setup_dismissed before navigating to /chat so generic first-run overlay does not immediately block exploration. Health diagnostics now labels missing URL, missing API key, invalid API key, unreachable server, and degraded feature checks separately; missing API key no longer renders the generic core-health outage panel. Diagnostics copy and visible raw response details redact secret-shaped keys before display/copy. Verification: focused Vitest passed 23 tests across option-setup-readiness, tldw-connection-status, health-status.design-system, and FirstRunGate. Repo-installed ESLint on touched UI files exited 0 with warnings only in existing health-status patterns and the known Next pages-directory notice when using the app config from apps/. git diff --check passed. Bandit not applicable because this stage touched TypeScript/TSX, docs, and Backlog task files only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 3 WebUI setup and health UX remediation is complete. /setup now exposes a self-host connection panel with server URL, password-masked API key, test connection, key-location help, and a skip/explore path that persists assistant_setup_dismissed before navigating to /chat. Health diagnostics now separates missing server URL, missing API key, invalid API key, unreachable server, and degraded feature checks, with missing credentials no longer shown as a generic core-health outage. Health diagnostics copy and raw response display redact secret-shaped fields. Verification: focused Vitest passed 23 tests; repo-installed ESLint exited 0 with warnings only; git diff --check passed. Bandit was not applicable because only TypeScript/TSX, docs, and Backlog task files changed.
<!-- SECTION:FINAL_SUMMARY:END -->
