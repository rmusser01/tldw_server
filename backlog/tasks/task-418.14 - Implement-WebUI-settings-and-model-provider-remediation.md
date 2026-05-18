---
id: TASK-418.14
title: Implement WebUI settings and model provider remediation
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-05-18 17:07'
labels:
  - ux
  - webui
  - extension
  - implementation
  - settings
  - models
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-17-webui-settings-models-implementation-plan.md
  - >-
    Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
  - >-
    Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
parent_task_id: TASK-418
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the WP5 settings and model/provider UX remediation slice for WebUI/extension. Scope: task-led settings grouping, provider-keys label repair, configured-first model/provider orientation, separation of routine preferences from data/destructive actions, prompt/settings route relationship guards, and focused unit/browser verification. Preserve existing routes, advanced controls, and backend APIs unless existing frontend data cannot responsibly represent the required UX state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Settings navigation has task-led groups and no visible dotted i18n keys.
- [ ] #2 /settings/provider-keys has a user-facing label and remains searchable/filterable from settings navigation.
- [ ] #3 Routine settings are separated from data/import/export/reset actions while preserving existing reset safeguards.
- [x] #4 /settings/model prioritizes default/configured/usable model setup before full catalog browsing while preserving advanced controls.
- [x] #5 Prompt Library, Prompt Studio, and Prompt Studio settings route intent remains distinct and covered by tests.
- [ ] #6 Focused Vitest settings/model tests, settings browser workflow checks, WP4 responsive landmarks, git diff --check, and applicable security checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Started implementation in clean worktree codex/webui-settings-models from origin/dev at e61681e99a04b655d14404d96f90f8f3b54b12aa after PR #1839 merged. Main checkout remains dirty and unrelated. Following Docs/superpowers/plans/2026-05-17-webui-settings-models-implementation-plan.md with TDD: add failing settings label/grouping tests before product code changes.

Completed first remediation slice: regrouped Settings navigation around user tasks (Connect, AI & Models, Experience, Knowledge & Workspace, Safety & Admin, About); fixed Provider Keys nav token; added locale guards for nav labels across source locales; regenerated public settings locale mirrors with the existing sync script.

Verification: bunx vitest run src/components/Layouts/__tests__/settings-nav.guardian.test.ts src/components/Layouts/__tests__/settings-layout-labels.test.tsx src/components/Layouts/__tests__/settings-layout-filter.test.tsx src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx src/components/Layouts/__tests__/settings-layout-active-route.test.ts -> 5 files / 29 tests passed.

Completed second remediation slice: split high-risk data management actions out of routine General Settings into /settings/data; preserved the existing import/export flow, Firefox private-mode sync, typed RESET confirmation, danger reset button, reload cancellation, and storage-clearing code path. Added settings nav Data Management group, source locale keys, and regenerated public settings locale mirrors.

Verification: bunx vitest run src/components/Option/Settings/__tests__/GeneralSettings.test.tsx src/components/Option/Settings/__tests__/DataManagementSettings.test.tsx src/components/Option/Settings/__tests__/system-settings.highlight-preview.test.ts src/components/Layouts/__tests__/settings-nav.guardian.test.ts src/components/Layouts/__tests__/settings-layout-filter.test.tsx src/components/Layouts/__tests__/settings-layout-focus-order.test.tsx src/components/Layouts/__tests__/settings-layout-active-route.test.ts -> 7 files / 31 tests passed. git diff --check passed. bunx tsc --noEmit --pretty false remains blocked by existing unrelated TypeScript baseline failures before touched settings files, starting in audio/chat/flashcards test fixtures.

Completed third remediation slice: /settings/model now puts default provider/model selection first, adds a provider readiness summary before the full catalog, and orders model choices configured-first while preserving the full AvailableModelsList catalog and OpenAI OAuth controls. The readiness summary uses server-returned chat models plus provider-key/OAuth status without adding backend API requirements.

Verification: bunx vitest run src/components/Option/Models/__tests__/ModelsBody.test.tsx src/components/Option/Models/__tests__/modelsDisplayUtils.test.ts src/components/Option/Models/__tests__/AvailableModelsList.test.tsx -> 3 files / 9 tests passed. git diff --check passed.

Completed fourth remediation slice: added browser route-intent guards for Prompt Library/Prompts workspace, legacy /prompt-studio redirect, /settings/prompt, and /settings/prompt-studio. No route/component code changes were needed; existing route ownership already matched the intended UX contract.

Verification: bunx playwright test e2e/workflows/tier-1-critical/settings-core.spec.ts --grep "Prompt route intent" --reporter=line -> 3 passed. bunx playwright test e2e/workflows/settings.spec.ts e2e/workflows/tier-1-critical/settings-core.spec.ts --reporter=line -> 59 passed. git diff --check passed.
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
