---
id: TASK-394.5
title: Harden Quick Ingest URL and file input validation
status: Done
assignee: []
created_date: '2026-05-16 00:44'
updated_date: '2026-05-29 03:29'
labels:
  - quick-ingest
  - ux
  - validation
  - task-5
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-16-quick-ingest-ux-remediation-implementation-plan.md
parent_task_id: TASK-394
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute implementation plan Task 5: strengthen URL/text/file input validation, duplicate prevention, unsupported content messaging, and truthful file-size handling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 URL and text validation prevents common invalid submissions with clear recovery copy
- [x] #2 Duplicate or unsupported content is detected or messaged consistently with backend limits
- [x] #3 File-size handling truthfully reflects the implemented browser memory/upload strategy
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Canonical completed record: `backlog/completed/task-394.5 - Harden-Quick-Ingest-URL-and-file-input-validation.md`. This `backlog/tasks/` file is a tracker mirror retained for PR visibility and should not be treated as a separate closeout record.

Latest origin/dev already contains the Task 5 Quick Ingest validation behavior. Verified normalized URL dedupe, mixed valid/invalid paste summary copy, supported file-type alignment, unsupported-file rejection, and the truthful 50 MB buffered-client upload limit.

Verification: `bun run test src/components/Common/QuickIngest/__tests__/AddContentStep.url-detection.test.ts src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx src/services/__tests__/quick-ingest-batch.test.ts --maxWorkers=1 --no-file-parallelism` passed 72 tests after `bun install` under `apps/` repaired copied worktree package links. Verification: `npx playwright test e2e/workflows/media-ingest.spec.ts --grep "quick ingest configure options stay reachable" --project=chromium --reporter=line` passed 1 test in 55.3s. Bandit is not applicable because this closeout branch only updates Backlog task metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Tracker mirror for the authoritative completed record at `backlog/completed/task-394.5 - Harden-Quick-Ingest-URL-and-file-input-validation.md`.

Closed Task 5 on latest origin/dev after verifying Quick Ingest validation and file-limit behavior are already implemented. Quick Ingest normalizes URLs for dedupe while preserving submitted/displayed URLs, summarizes mixed valid and invalid URL paste results, aligns file support copy with picker/detection behavior, rejects unsupported local file types earlier, and presents the current 50 MB browser-buffered upload limit.

Verification: `bun run test src/components/Common/QuickIngest/__tests__/AddContentStep.url-detection.test.ts src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx src/services/__tests__/quick-ingest-batch.test.ts --maxWorkers=1 --no-file-parallelism` passed 72 tests. `npx playwright test e2e/workflows/media-ingest.spec.ts --grep "quick ingest configure options stay reachable" --project=chromium --reporter=line` passed 1 test in 55.3s. Bandit is not applicable because this closeout branch only changes Backlog task metadata.
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
