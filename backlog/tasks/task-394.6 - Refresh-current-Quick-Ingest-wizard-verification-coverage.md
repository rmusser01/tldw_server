---
id: TASK-394.6
title: Refresh current Quick Ingest wizard verification coverage
status: Done
assignee: []
created_date: '2026-05-16 00:44'
updated_date: '2026-05-29 03:54'
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
Canonical completed record: `backlog/completed/task-394.6 - Refresh-current-Quick-Ingest-wizard-verification-coverage.md`. This `backlog/tasks/` file is a tracker mirror retained for PR visibility and should not be treated as a separate closeout record.

Latest origin/dev already contains the Task 6 Quick Ingest coverage refresh. During closeout verification, the full WebUI Quick Ingest sweep exposed one stale helper assertion: `assertQuickIngestCompletedResults` accepted the older short summary shape but not the current full summary with skipped, not submitted, failed, and cancelled counts. Updated the helper regex to accept the current summary while preserving the older shorter shape.

Verification: `bun run test src/components/Common/QuickIngest/__tests__ src/services/__tests__/quick-ingest-batch.test.ts src/services/__tests__/quick-ingest-session-reattach.test.ts --maxWorkers=1 --no-file-parallelism` passed 208 tests after `bun install` under `apps/` repaired copied worktree package links. Verification: `npx playwright test e2e/workflows/media-ingest.spec.ts --grep "quick ingest restores skipped duplicate URL results after reopen" --project=chromium --reporter=line` passed 1 test in 48.5s. Verification: `npx playwright test e2e/workflows/media-ingest.spec.ts --grep "Quick Ingest" --project=chromium --reporter=line` passed 13 tests in 4.4m after rebasing onto latest origin/dev. Bandit is not applicable because this closeout branch only updates frontend Playwright helper code and Backlog metadata.

Known verification gap retained from the completed record: focused extension Playwright is still blocked by the extension build/globalSetup harness before tests start, not by the Task 6 migrated specs.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Tracker mirror for the authoritative completed record at `backlog/completed/task-394.6 - Refresh-current-Quick-Ingest-wizard-verification-coverage.md`.

Closed Task 6 on latest origin/dev after verifying current WebUI Quick Ingest coverage and fixing one stale WebUI helper assertion for the current results summary copy. Shared Quick Ingest Vitest coverage and the full WebUI Quick Ingest Playwright sweep now pass. The extension harness caveat remains documented in the authoritative completed record.
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
