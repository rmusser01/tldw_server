---
id: TASK-394.6
title: Refresh current Quick Ingest wizard verification coverage
status: Done
assignee: []
created_date: '2026-05-16 00:44'
updated_date: '2026-05-16 04:22'
labels:
  - quick-ingest
  - tests
  - task-6
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-16-quick-ingest-ux-remediation-implementation-plan.md
parent_task_id: TASK-394
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute implementation plan Task 6: update or replace stale selectors/tests and add focused coverage for the active wizard states touched by this remediation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current active wizard tests cover default, URL/text/file, validation, success/failure, and close/cancel paths where feasible
- [x] #2 Legacy selector tests are removed, renamed, or clearly quarantined if stale
- [x] #3 Verification commands for the touched UI slice pass or have documented environmental blockers
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Task 6 after closing TASK-394.5. Scope: refresh current Quick Ingest wizard e2e selectors/coverage, classify extension quick-ingest specs against the active-path map, run focused shared/WebUI/extension verification where available, and document any harness blockers.

Implemented in be1984cee. WebUI journey helpers now target the active wizard dialog and current Add/Configure/Review/Processing/Results controls instead of stale quick-ingest-run or old results-panel fallbacks. WebUI e2e now asserts current first-open purpose/50 MB copy and mixed valid/invalid URL paste validation. The targeted extension quick-ingest UX audit and cancel specs are classified as active-wizard specs and now use current wizard roles, current supported-format copy, wizard-results-step, completed/error regions, Use defaults & process, and Cancel All.

Verification: ./node_modules/.bin/vitest run src/components/Common/QuickIngest/__tests__ src/services/__tests__/quick-ingest-batch.test.ts src/services/__tests__/quick-ingest-session-reattach.test.ts --maxWorkers=1 --no-file-parallelism passed with 15 files / 178 tests. WebUI Playwright passed outside the macOS sandbox: TLDW_WEB_AUTOSTART=false TLDW_WEB_URL=http://127.0.0.1:18001 NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY bunx playwright test e2e/workflows/media-ingest.spec.ts --grep "Quick Ingest" --project=chromium --reporter=line; 11 tests passed. git diff --check passed.

Known verification gap: focused extension Playwright did not reach test execution. Command: bunx playwright test tests/e2e/quick-ingest-ux-audit.spec.ts tests/e2e/quick-ingest-cancel.spec.ts --reporter=line. Global setup tried npm run build:chrome:prod and failed with /bin/sh: npm: command not found. The fallback bun run build:chrome:prod entered wxt build, emitted duplicate-import warnings, then stayed silent/hung for several minutes until the test/build processes were terminated; final output was error: script "build:chrome:prod" exited with code 1. Targeted TypeScript check of the edited extension specs reported no direct errors in those specs after fixes, but failed in imported baseline helpers: tests/e2e/utils/extension-build.ts browser/argument-count errors and tests/e2e/utils/extension-id.ts Page/Worker concat typing. Bandit skipped because touched files are frontend Playwright TypeScript only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 6 complete with an extension harness caveat: current WebUI Quick Ingest coverage now targets the active wizard flow and passes focused shared/WebUI verification. The targeted extension specs were migrated away from stale legacy selectors, but their Playwright execution remains blocked by the extension build/globalSetup harness before tests start.
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
