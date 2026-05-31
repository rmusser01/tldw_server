---
id: TASK-418.14
title: Implement WebUI settings and model provider remediation
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-18 19:49'
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
- [x] #1 Settings navigation has task-led groups and no visible dotted i18n keys.
- [x] #2 /settings/provider-keys has a user-facing label and remains searchable/filterable from settings navigation.
- [x] #3 Routine settings are separated from data/import/export/reset actions while preserving existing reset safeguards.
- [x] #4 /settings/model prioritizes default/configured/usable model setup before full catalog browsing while preserving advanced controls.
- [x] #5 Prompt Library, Prompt Studio, and Prompt Studio settings route intent remains distinct and covered by tests.
- [x] #6 Focused Vitest settings/model tests, settings browser workflow checks, WP4 responsive landmarks, git diff --check, and applicable security checks are recorded.
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

Final verification 2026-05-18: focused UI Vitest suite passed 11 files / 41 tests; settings Playwright workflow pair passed 59 tests; WP4 responsive landmarks passed 12 tests including /settings and /settings/model; git diff --check passed. Documentation governance scans for the child plan and TASK-418.2 produced no placeholder/trailing-whitespace/non-ASCII findings and diff-check passed. Bandit was not run because this slice touched frontend TypeScript/TSX, Playwright tests, Markdown, locale JSON, and Backlog files only. Full apps/packages/ui TypeScript remains blocked by pre-existing repo-wide baseline debt outside this slice; first failures are in audio, chat composer, common prompt utils, quick-ingest, flashcards, playground, services, and route baseline tests before this branch’s settings/model files.

Draft PR opened: https://github.com/rmusser01/tldw_server/pull/1845

PR #1845 review follow-up 2026-05-18: Gemini flagged the reset reload timer in DataManagementSettings because handleResetAll uses a bare setTimeout while the component cleanup and import cancellation use reloadTimeoutRef. Reopening the task to route the reset reload timer through the existing ref and verify the focused settings tests.

PR #1845 review follow-up verification 2026-05-18: added a regression test proving the reset reload timer is cleared on DataManagementSettings unmount, then stored the reset timer in reloadTimeoutRef and nulled it before window.location.reload. Targeted test passed before broad rerun: bunx vitest run src/components/Option/Settings/__tests__/DataManagementSettings.test.tsx -> 1 file / 2 tests passed. Broader focused settings/model Vitest suite passed 11 files / 42 tests. git diff --check passed. bunx tsc --noEmit --pretty false remains blocked by existing repo-wide UI baseline errors outside this touched settings slice, with no touched settings file diagnostics in the observed output.

PR #1845 second review sweep 2026-05-18: CodeRabbit/Qodo added actionable comments after commit c229a70aa. Current items to verify/fix: locale directory filtering in settings nav guardian, provider-key error handling in model readiness UI, keyboard-accessible import trigger, exact prompt settings URL assertion, duplicate FINAL_SUMMARY markers in TASK-418.14 and TASK-418.2, raw syncFirefoxData error logging, and model usability/configuration derived from server catalog fields.

PR #1845 second review sweep verification 2026-05-18: fixed the new CodeRabbit/Qodo comments by filtering locale guard iteration to directories, showing provider-key load failures separately from configured counts, deriving model configured/usable state from provider keys plus server model flags, replacing the import label with a disabled-aware button that clicks the hidden file input, removing raw syncFirefoxData error logging, tightening the prompt settings URL assertion, and removing duplicate FINAL_SUMMARY markers from TASK-418.14 and TASK-418.2. Verification: affected Vitest tests passed 3 files / 21 tests; broader focused settings/model Vitest suite passed 11 files / 45 tests; git diff --check passed. Focused Playwright prompt-route command was rerun outside the sandbox after a port-bind EPERM, but skipped 3 tests because the E2E fixture marked the backend server unavailable. bunx tsc --noEmit --pretty false still fails on existing repo-wide UI baseline diagnostics outside this touched slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed WP5 settings/model-provider UX remediation for WebUI/extension. Implemented task-led settings grouping, user-facing Provider Keys navigation, separate Data Management settings for import/export/reset, configured-first /settings/model defaults and provider readiness, model utility coverage, and browser route-intent guards for /prompts, /prompt-studio, /settings/prompt, and /settings/prompt-studio. Verification passed for focused Vitest settings/model coverage, settings Playwright workflows, WP4 responsive landmarks, and diff/governance checks. Full TypeScript remains blocked by existing unrelated baseline debt; Bandit is not applicable to this frontend/docs-only slice.

PR review follow-up: addressed Gemini's reset reload timeout comment by routing the reset reload through reloadTimeoutRef so import cancellation and unmount cleanup can clear it. Added focused regression coverage and re-ran the focused settings/model suite.

Second PR review sweep: addressed all still-actionable CodeRabbit/Qodo comments available at the time of refresh, including import accessibility, model readiness correctness, provider-key error state, guard/test hardening, raw error logging, and Backlog marker cleanup.
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
