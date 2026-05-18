---
id: TASK-418.2
title: Plan WebUI settings and model provider implementation
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-18 18:03'
labels:
  - ux
  - design
  - webui
  - extension
  - planning
  - settings
  - models
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
  - >-
    Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
  - >-
    Docs/superpowers/plans/2026-05-17-webui-settings-models-implementation-plan.md
parent_task_id: TASK-418
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Documentation-only child implementation plan for the approved WebUI/extension UX remediation program Task 5. Scope maps findings F5, F7, F11, F16, F15 support, and F2 support into a reviewable settings navigation, model/provider readiness, and destructive-action separation implementation plan without product code changes in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Child implementation plan exists for the settings and model/provider remediation slice.
- [x] #2 Plan stays documentation-only and does not modify product frontend or backend code.
- [x] #3 Plan maps the slice to F5, F7, F11, F16, F15 support, and F2 support from the approved UX remediation program.
- [x] #4 Plan defines task-led settings groups and preserves existing route paths unless a focused data-management route is explicitly selected during implementation.
- [x] #5 Plan names settings nav, locale, general settings, provider keys, model settings, model catalog, unit-test, and browser-test ownership.
- [x] #6 Plan requires configured-first model/provider UX while preserving full catalog and advanced controls.
- [x] #7 Planning verification commands and results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Created Docs/superpowers/plans/2026-05-17-webui-settings-models-implementation-plan.md.
- Reused the approved parent plan and remediation spec as the source of scope.
- Route rows covered by the plan are /settings, /settings/tldw, /settings/provider-keys, /settings/model, /login, /privileges, /prompts, /prompt-studio, and settings subroutes.
- The plan starts with visible label, grouping, model ordering, and destructive-action separation tests before implementation.
- The plan preserves ProviderKeysSettings, ModelsBody, AvailableModelsList, ModelSettings, SettingsLayout, and the existing route registry patterns.
- Verification run for this planning artifact: placeholder-language scan exited 1 with no output; ASCII/trailing-whitespace scan exited 1 with no output; git diff check exited 0; Node coverage check confirmed required route, finding, file, and test tokens are present.
- Bandit was not run because this task changed only Markdown planning and Backlog task files.

Implementation follow-through recorded 2026-05-18: child task TASK-418.14 completed the settings/model-provider implementation slice from this plan in branch codex/webui-settings-models. Scope included task-led settings grouping, Provider Keys label repair, Data Management separation, configured-first /settings/model orientation, prompt route-intent browser guards, focused Vitest settings/model tests, settings Playwright workflow tests, and WP4 responsive landmark verification. Full apps/packages/ui TypeScript remains blocked by pre-existing repo-wide baseline debt outside this slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the documentation-only child implementation plan for WP5 settings and model/provider remediation. The plan defines route scope, settings grouping, provider-key label repair, configured-first model UX, destructive-action separation, tests, and browser QA gates without changing product code.
<!-- SECTION:FINAL_SUMMARY:END -->

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
