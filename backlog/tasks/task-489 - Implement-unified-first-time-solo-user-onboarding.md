---
id: TASK-489
title: Implement unified first-time solo user onboarding
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-05-31 06:14'
labels: []
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-05-31-first-time-solo-user-onboarding-prd-design.md
  - >-
    Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
  - >-
    backlog/tasks/task-488 -
    Plan-unified-first-time-solo-user-onboarding-implementation.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Parent implementation task for the approved unified first-time solo-user onboarding plan. Coordinate child slices for backend setup state/access, provider setup, first-chat completion, WebUI wizard, docs/CLI cleanup, and final E2E/security verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All child implementation slices are completed or explicitly deferred with rationale
- [ ] #2 Final verification checklist from the implementation plan is run or documented with blockers
- [ ] #3 Bandit is run on touched backend scope before final completion
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created child implementation tasks: TASK-490 backend state/access foundation, TASK-491 provider catalog/validation, TASK-492 first-chat completion/settings endpoints, TASK-493 frontend setup client/shell, TASK-494 progressive wizard steps, TASK-495 docs/CLI cleanup, TASK-496 final E2E/security verification. Baseline checks before implementation: setup remote admin pytest 8 passed; config provider endpoint pytest 33 passed; OnboardingConnectForm design-system Vitest 4 passed. Bun install in apps initially hung in extension wxt postinstall and was stopped after dependencies were linked; targeted Vitest then passed.

TASK-491 completed after spec and code-quality review. Provider setup now covers catalog/save/validation behavior, runtime config mapping, cache refresh, first-run write gating, local endpoint target guarding, and config-section insertion regressions.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
