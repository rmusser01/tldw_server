---
id: TASK-13113
title: Fix Agent Task Jobs consumer missing-definition crash
status: To Do
created_date: 2026-08-24 06:08
labels:
- scheduled-tasks
- agent-task
- bug
- jobs
- phase-4d-dependency
priority: high
references:
- tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py
- tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py
- tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The existing Agent Task Jobs consumer creates a normalized scheduled-task run before checking whether the referenced definition exists. The storage contract rejects a missing definition, so a stale/deleted-definition Job raises KeyError instead of returning the documented skipped outcome. Fix this test-first without broadening Phase 4D product scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Job referencing a deleted or never-created definition terminates deterministically without crashing the worker.
- [ ] #2 The resulting behavior has an explicit observable outcome consistent with run-storage constraints; if no run can legally exist, the consumer response and audit/metrics behavior document that exception.
- [ ] #3 Existing run-slot dedupe, lifecycle, notification, health, timeout, and error behavior remains unchanged.
- [ ] #4 The focused Agent Task consumer suite and adjacent Scheduled Tasks automation tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
