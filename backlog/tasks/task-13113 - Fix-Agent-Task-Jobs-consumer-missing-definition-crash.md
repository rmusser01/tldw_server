---
id: TASK-13113
title: Fix Agent Task Jobs consumer missing-definition crash
status: Done
assignee: []
created_date: '2026-08-24 06:08'
updated_date: '2026-08-24 20:05'
labels:
  - scheduled-tasks
  - agent-task
  - bug
  - jobs
  - phase-4d-dependency
dependencies: []
references:
  - tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py
  - tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py
  - tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py
documentation:
  - >-
    Docs/superpowers/plans/2026-08-24-scheduled-tasks-phase4d-prerequisite-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The existing Agent Task Jobs consumer creates a normalized scheduled-task run before checking whether the referenced definition exists. The storage contract rejects a missing definition, so a stale/deleted-definition Job raises KeyError instead of returning the documented skipped outcome. Fix this test-first without broadening Phase 4D product scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Job referencing a deleted or never-created definition terminates deterministically without crashing the worker.
- [x] #2 The resulting behavior has an explicit observable outcome consistent with run-storage constraints; if no run can legally exist, the consumer response and audit/metrics behavior document that exception.
- [x] #3 Existing run-slot dedupe, lifecycle, notification, health, timeout, and error behavior remains unchanged.
- [x] #4 The focused Agent Task consumer suite and adjacent Scheduled Tasks automation tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented test-first in the isolated codex/scheduled-tasks-phase4d-agent-task-design worktree based on origin/dev 2c6553c4ed. RED: missing and cross-owner definitions both raised KeyError from create_scheduled_task_run before the fix. GREEN: the final four-file scheduled-task matrix passed 108 tests. Bandit scanned agent_task_jobs.py with zero findings. Worker review confirmed every returned result reaches complete_job(), while only raised exceptions reach fail_job(). Independent code review found no Critical or Important issues; its one Minor test-fixture precision finding was addressed before the final test run. TASK-13122 separately repaired a stale dev-baseline DefinitionRow test helper so the adjacent gate could execute. No known blockers or skipped required checks. Two unrelated untracked Watchlists templates remain intentionally excluded.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved the owner-scoped definition lookup ahead of run and notification database creation. Missing, deleted, or cross-owner definition IDs now produce the same explicit skipped Job result with run_id=None and reason=definition_missing, plus a bounded structured warning. No invalid run, notification, definition audit, or executor call is created. Existing valid-definition execution behavior is unchanged, and focused tests now enforce the exact result, side-effect, secrecy, and ownership contracts.
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
