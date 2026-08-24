---
id: TASK-13113
title: Fix Agent Task Jobs consumer missing-definition crash
status: Done
assignee: []
created_date: 2026-08-24 06:08
updated_date: 2026-08-25 00:34
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
- Docs/superpowers/plans/2026-08-24-scheduled-tasks-phase4d-prerequisite-implementation-plan.md
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented test-first on the rebased codex/scheduled-tasks-phase4d-agent-task-design branch at origin/dev 4091735b6f. PR #2816 review remediation restored terminal run-slot dedupe before missing-definition handling, added a defensive owner match for injected repositories, made definition/user/job identifiers visible in the rendered warning without logging payload data, and added typed/docstring-compliant regressions. RED runs reproduced the dedupe and owner-leak failures; GREEN verification passed the final four-file Scheduled Tasks matrix with 110 tests. Bandit reported zero findings, Ruff import-order checks passed, git diff --check passed, and independent review found no Critical or Important issues. Qodo's four inline findings were addressed. Two unrelated untracked Watchlists templates remain intentionally excluded.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The consumer now lets the existing storage boundary resolve an owner-matched terminal run before checking definition availability, preserving idempotent redelivery and recorded history. When no run exists, or an injected repository returns another owner's run, the Job returns the concealed side-effect-free definition_missing result with run_id=None. Missing-definition warnings render only definition_id, user_id, and job_id; focused tests verify the rendered output, owner isolation, terminal dedupe, and no invalid run, audit, notification, or executor call. The implementation plan and Phase 4D design now document the corrected dedupe-first contract.
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
