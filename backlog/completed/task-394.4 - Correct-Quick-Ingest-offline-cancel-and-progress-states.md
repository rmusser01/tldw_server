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
Canonical record: this `backlog/completed/` file is the authoritative final closeout for TASK-394.4. The active-tracker mirror at `backlog/tasks/task-394.4 - Correct-Quick-Ingest-offline-cancel-and-progress-states.md` points here and should not be treated as a separate task.

Started Task 4 in quick-ingest UX remediation branch. Scope: offline/disconnected processing guard, cancel/close distinction, neutral progress copy, and minimized widget terminal-state accuracy.

Implemented offline processing guards in Add and Review steps using the shared connection store, added retry recovery affordances, neutralized global processing copy, and split minimized widget terminal states into Done, Failed, Cancelled, and Interrupted. Functional commit: 9958abdc8.

Closeout verification on latest origin/dev confirmed the Task 4 behavior is still present: Add step disables quick processing while disconnected, shows server-offline recovery copy with retry, still allows Configure for queued items, and guards handleQuickProcess while offline/checking. ProcessingStep uses neutral global copy (`Processing and indexing content`) and FloatingProgressWidget splits Done, Failed, Cancelled, and Interrupted minimized terminal states.

Latest verification: `bun run test src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx src/components/Common/QuickIngest/__tests__/FloatingProgressWidget.test.tsx --maxWorkers=1 --no-file-parallelism` passed 61 tests after `bun install` under `apps/` repaired copied worktree package links. `npx playwright test e2e/workflows/media-ingest.spec.ts --grep "quick ingest can be dismissed during processing" --project=chromium --reporter=line` passed 1 test in 57.1s. Bandit is not applicable to the closeout PR because it only updates Backlog task metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 4 complete. This `backlog/completed/` file is the authoritative closeout. Offline/disconnected users can still add and configure items, but processing actions are blocked with explicit recovery copy and Retry connection. Review prevents final processing while disconnected. Progress copy uses neutral processing/indexing language. Minimized progress widget distinguishes completed, failed, cancelled, and interrupted sessions with non-success terminal states.

Latest verification on origin/dev: focused Quick Ingest Vitest coverage passed 61 tests, focused Playwright dismiss/resume coverage passed 1 test in 57.1s, and `git diff --check` passed. Bandit was skipped for the metadata-only closeout PR.
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
