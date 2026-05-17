---
id: TASK-421
title: Plan WebUI setup and connection flow implementation
status: Done
labels:
- ux
- design
- webui
- extension
- planning
- setup
priority: high
modified_files:
- Docs/superpowers/plans/2026-05-17-webui-setup-connection-flow-implementation-plan.md
- backlog/tasks/task-421 - Plan-WebUI-setup-and-connection-flow-implementation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the child implementation plan for the approved WebUI/extension UX remediation Task 3 first-run setup and connection-flow slice. The plan must be documentation-only, preserve existing setup/product intent, define the home/setup/auth/account state matrix, name route/component/test ownership, and map the slice to findings F3, F15 support, and F1 support.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Child implementation plan exists for the first-run setup and connection-flow remediation slice.
- [x] Plan stays documentation-only and does not modify product frontend or backend code.
- [x] Plan maps the slice to F3, F15 support, and F1 support from the approved UX remediation program.
- [x] Plan defines the `/`, `/setup`, auth/account, hosted-placeholder, redirect, and 404 state matrix.
- [x] Plan names route, component, placeholder, redirect, connection-store, unit-test, and browser-test ownership.
- [x] Plan preserves existing self-host and hosted product intent instead of inventing private hosted features.
- [x] Planning verification commands and results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created `Docs/superpowers/plans/2026-05-17-webui-setup-connection-flow-implementation-plan.md`.
- Reused the approved parent plan and remediation spec as the source of scope.
- Route rows covered by the plan are `/`, `/setup`, `/login`, `/signup`, `/account`, `/profile`, `/privileges`, `/config`, `/billing`, and `/404`.
- Related shared placeholder routes included in the implementation scope are `/billing/success`, `/billing/cancel`, `/auth/reset-password`, `/auth/magic-link`, and `/auth/verify-email`.
- The plan preserves existing `option-index.tsx`, `option-setup.tsx`, `OnboardingConnectForm`, `RoutePlaceholder`, and `RouteRedirect` ownership rather than proposing a replacement setup system.
- Verification run for this planning artifact:
  - Placeholder-language scan against the plan and task files exited 1 with no output.
  - ASCII/trailing-whitespace scan against the plan and task files exited 1 with no output.
  - `git diff --check -- Docs/superpowers/plans/2026-05-17-webui-setup-connection-flow-implementation-plan.md "backlog/tasks/task-421 - Plan-WebUI-setup-and-connection-flow-implementation.md"` exited 0.
  - `node -e` coverage check confirmed the required route, finding, component, and test tokens are present.
- Bandit was not run because this task changed only Markdown planning and Backlog task files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the documentation-only child implementation plan for WebUI/extension first-run setup and connection-flow remediation. The plan defines the setup/auth/account state matrix, route/component/test ownership, and browser QA gates without changing product code.
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
