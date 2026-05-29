---
id: TASK-394.4
title: Correct Quick Ingest offline cancel and progress states
status: Done
assignee: []
created_date: '2026-05-16 00:43'
updated_date: '2026-05-29 01:59'
labels:
  - quick-ingest
  - ux
  - task-4
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-16-quick-ingest-ux-remediation-implementation-plan.md
parent_task_id: TASK-394
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute implementation plan Task 4: align offline checks, cancel/close behavior, in-flight processing, progress copy, and background status with real system state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Offline and network failure states surface before or during submit with actionable recovery
- [x] #2 Cancel/close behavior distinguishes draft dismissal from in-flight processing
- [x] #3 Progress/background status copy does not imply unsupported background jobs or hidden completion tracking
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Latest origin/dev already contains the Task 4 Quick Ingest recovery behavior. Verified Add step disables quick processing while disconnected, shows server-offline recovery copy with retry, still allows Configure for queued items, and guards handleQuickProcess while offline/checking. Verified ProcessingStep uses neutral global copy (`Processing and indexing content`) and FloatingProgressWidget splits Done, Failed, Cancelled, and Interrupted minimized terminal states.

Verification: `bun run test src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx src/components/Common/QuickIngest/__tests__/FloatingProgressWidget.test.tsx --maxWorkers=1 --no-file-parallelism` passed 61 tests after `bun install` under `apps/` repaired copied worktree package links. Verification: `npx playwright test e2e/workflows/media-ingest.spec.ts --grep "quick ingest can be dismissed during processing" --project=chromium --reporter=line` passed 1 test in 57.1s. Bandit is not applicable because this closeout branch only updates Backlog task metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed Task 4 on latest origin/dev after verifying the offline, cancel, progress, and minimized background states are already implemented. Add step processing is blocked while disconnected with visible retry/recovery copy, configuration remains available, QuickIngestWizardModal guards quick processing while offline/checking, ProcessingStep uses neutral global progress copy, and FloatingProgressWidget distinguishes Done, Failed, Cancelled, and Interrupted terminal states.

Verification: `bun run test src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx src/components/Common/QuickIngest/__tests__/FloatingProgressWidget.test.tsx --maxWorkers=1 --no-file-parallelism` passed 61 tests. `npx playwright test e2e/workflows/media-ingest.spec.ts --grep "quick ingest can be dismissed during processing" --project=chromium --reporter=line` passed 1 test in 57.1s. Bandit is not applicable because this closeout branch only changes Backlog task metadata.
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
