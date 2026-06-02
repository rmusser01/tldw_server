---
id: TASK-511
title: 'Task 5: Add dedicated onboarding UAT Playwright fixtures and config'
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 04:49'
labels:
  - onboarding-uat
  - playwright
  - test
dependencies: []
priority: medium
modified_files:
  - apps/tldw-frontend/e2e/onboarding-uat/playwright.config.ts
  - apps/tldw-frontend/e2e/onboarding-uat/fixtures.ts
  - apps/tldw-frontend/e2e/onboarding-uat/scenarios.ts
  - apps/tldw-frontend/e2e/onboarding-uat/helpers.ts
  - apps/tldw-frontend/__tests__/e2e-harness-readiness.guard.test.ts
  - apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the dedicated onboarding UAT Playwright config, fixtures, scenario metadata, UI helpers, and guard coverage that keeps the UAT specs on real backend/provider behavior instead of route-mocked provider paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Playwright config exists for onboarding UAT with desktop/mobile projects and no webServer
- [x] #2 Fixtures provide clean first-run page setup, diagnostics capture, and artifact helpers without seeding setup completion
- [x] #3 Scenario metadata is JSON-compatible and supports Tier A scenario filtering
- [x] #4 UI helpers cover opening setup, connecting single-user, sending first chat, step capture, and diagnostics assertions
- [x] #5 Guard test prevents onboarding UAT specs from using page.route provider mocks, seedAuth, setup-completion storage flags, or waitForTimeout
- [x] #6 Vitest guard verification and Bandit skip for non-Python touched scope are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the dedicated onboarding UAT Playwright harness layer. Added a standalone Playwright config with desktop/mobile projects and no webServer because the runner owns services; added clean first-run fixtures that grant clipboard permissions, clear browser storage without seeding auth/setup completion, capture console/page/request diagnostics, and write diagnostics to the runner artifact root; added JSON-compatible Tier A scenario metadata; added helpers for opening first-run setup, connecting single-user setup, sending first chat through the existing ChatPage/stream helpers, capturing step screenshots/JSON, and asserting critical diagnostics. Extended the readiness guard so Task 5 harness files and any later onboarding UAT spec files cannot use page.route provider mocks, seedAuth, setup-completion storage flags, or waitForTimeout. Fixed an existing Stage 4 guard violation by replacing direct networkidle/fixed sleep settling with waitForVisualSettle and removing an unused helper exposed by the touched-file lint pass. Verification: guard red failed on missing onboarding UAT config before implementation; `bunx vitest run __tests__/e2e-harness-readiness.guard.test.ts` passed with 16 tests; `bunx eslint e2e/onboarding-uat __tests__/e2e-harness-readiness.guard.test.ts e2e/smoke/stage4-axe-high-risk-routes.spec.ts` exited 0 with no warnings; narrow TypeScript compile over the new UAT files exited 0; `git diff --check` passed. Repo-wide `bunx tsc --noEmit --pretty false --project tsconfig.json` remains blocked by existing shared-package/module-resolution and prior JS-module typing baseline errors unrelated to the new UAT files. `bunx playwright test -c e2e/onboarding-uat/playwright.config.ts --list` loads the config but exits 1 with `No tests found`, expected until Task 6 adds specs. Bandit skipped because Task 5 touched TypeScript/test/task files only; no Python code changed.
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
