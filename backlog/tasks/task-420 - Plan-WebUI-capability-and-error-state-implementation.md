---
id: TASK-420
title: Plan WebUI capability and error state implementation
status: Done
labels:
- ux
- design
- webui
- extension
- planning
- states
priority: high
modified_files:
- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md
- backlog/tasks/task-420 - Plan-WebUI-capability-and-error-state-implementation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the child implementation plan for the approved WebUI/extension UX remediation Task 2 capability and error-state slice. The plan must be documentation-only, reuse existing shared state primitives before adding new ones, define the capability vocabulary, name route adopters and tests, and map the slice to findings F4, F5 support, F9, and F18 support.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Child implementation plan exists for the capability/error-state remediation slice.
- [x] Plan stays documentation-only and does not modify product frontend or backend code.
- [x] Plan maps the slice to F4, F5 support, F9, and F18 support from the approved UX remediation program.
- [x] Plan defines the capability vocabulary and maps it to existing design-system state keys.
- [x] Plan names first adopter routes, later route-family adopters, component ownership, test ownership, and browser QA expectations.
- [x] Plan requires raw endpoint/status details to move behind diagnostics rather than disappear.
- [x] Planning verification commands and results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created `Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md`.
- Reused the approved parent plan and remediation spec as the source of scope instead of introducing new product scope.
- First implementation adopters are `/sources`, `/scheduled-tasks`, and `/integrations`.
- Later adopters are explicitly listed for `/admin`, `/agents`, `/agent-tasks`, `/acp-playground`, `/settings/model`, `/evaluations`, `/mcp-hub`, `/skills`, `/tts`, `/speech`, and `/data-tables`.
- Shared state foundation is constrained to existing `StatePanel`, `RecoveryCallout`, `DiagnosticRow`, `RouteErrorBoundary`, and `BackendUnavailableRecovery` patterns before any helper is added.
- Verification run for this planning artifact:
  - Placeholder-language scan against the plan and task files exited 1 with no output.
  - ASCII/trailing-whitespace scan against the plan and task files exited 1 with no output.
  - `git diff --check -- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md "backlog/tasks/task-420 - Plan-WebUI-capability-and-error-state-implementation.md"` exited 0.
  - `node -e` coverage check confirmed the required route, finding, component, and test tokens are present.
- Bandit was not run because this task changed only Markdown planning and Backlog task files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the documentation-only child implementation plan for WebUI/extension capability and error-state remediation. The plan defines the capability vocabulary, locks first-adopter route scope, names shared state/component/test ownership, and records verification without modifying product code.
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
